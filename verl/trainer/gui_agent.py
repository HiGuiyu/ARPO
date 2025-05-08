import ast
import re
import math
from PIL import Image
import ray
import os
from io import BytesIO
import base64
import datetime
import traceback

from desktop_env.desktop_env import DesktopEnv


steve_system_prompt = '''\
You are STEVE, an AI that completes tasks on a user's computer with mouse and keyboard operations. Your role as an assistant involves thoroughly exploring the user task through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of screenshot analysis, action reflection, task planning and error backtracing to complete the user task step-by-step. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> thought <|end_of_thought|> Each step should include detailed considerations such as screenshot analysis, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {thoughts, rationale, decision, python(optional)} <|end_of_solution|>"""

You can use the following functions:
```python
computer.mouse.move(<|object_ref_start|>"target_element"<|object_ref_end|><|point_start|>(x,y)<|point_end|>)
computer.mouse.single_click()
computer.mouse.double_click()
computer.mouse.right_click() 
computer.mouse.scroll(dir="up/down")
computer.mouse.drag(<|object_ref_start|>"target_element"<|object_ref_end|><|point_start|>(x,y)<|point_end|>) # drag from current cursor position to the element

computer.keyboard.write("text")
computer.keyboard.press("key") # press a key once (donw and up)
computer.keyboard.keyDown("key") # press and hold a key
computer.keyboard.keyUp("key") # release a key
computer.keyboard.hotkey("key1", "key2", ...) # press multiple keys simultaneously to trigger a shortcut

```
'''


uitars_system_prompt = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Use English in `Thought` and `Action` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

# 定义一个函数来解析每个 action
def parse_action(action_str):
    try:
        # 解析字符串为 AST 节点
        node = ast.parse(action_str, mode='eval')

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 获取函数名
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # 获取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值，这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {
            'function': func_name,
            'args': kwargs
        }

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None

def escape_single_quotes(text):
    # 匹配未转义的单引号（不匹配 \\'）
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def linear_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    if width * height > max_pixels:
        """
        如果图片超过/低于像素限制，则计算一个缩放因子resize_factor，使图片的像素数缩小到等于或小于max_pixels。这个缩放因子是通过开平方根计算的，确保纵横比保持不变,这样原始的相对坐标可以不经转换直接复用
        """
        resize_factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * resize_factor), int(height * resize_factor)
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        width, height = math.ceil(width * resize_factor), math.ceil(height * resize_factor)

    return height, width 

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar



def parse_action_to_structure_output(text, factor, origin_resized_height, origin_resized_width, model_type, max_pixels=16384*28*28, min_pixels=100*28*28):
    text = text.strip()
    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(origin_resized_height, origin_resized_width, factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels)

    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    assert "Action:" in text
    action_str = text.split("Action:")[-1]

    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        elif "finished(content" in action_str:
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content
            
            # 使用正则表达式进行替换
            pattern = r"finished\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "finished(content='" + action_str + "')"
            
        all_action.append(action_str)

    parsed_actions = [parse_action(action.replace("\n","\\n").lstrip()) for action in all_action]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance == None:
            print(f"Action can't parse: {raw_str}")
            raise ValueError(f"Action can't parse: {raw_str}") 
        action_type = action_instance["function"]
        params = action_instance["args"]

        # import pdb; pdb.set_trace()
        action_inputs = {}
        for param_name, param in params.items():
            if param == "": continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param
            
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and split the string by commas
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                # Convert to float and scale by 1000
                # Qwen2.5vl output absolute coordinates, qwen2vl output relative coordinates
                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(float(num/smart_resize_height))
                        else:
                            float_numbers.append(float(num/smart_resize_width))
                else:
                    float_numbers = [float(num) / factor for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                action_inputs[param_name.strip()] = str(float_numbers)

        # import pdb; pdb.set_trace()
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions

def parsing_response_to_pyautogui_code(responses, image_height: int, image_width:int, input_swap:bool=True) -> str:
    '''
    将M模型的输出解析为OSWorld中的action，生成pyautogui代码字符串
    参数:
        response: 包含模型输出的字典，结构类似于：
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }
    返回:
        生成的pyautogui代码字符串
    '''

    pyautogui_code = f"import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""
        
        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(1)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})
        
        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"
            
            elif hotkey == "arrowup":
                hotkey = "up"
            
            elif hotkey == "arrowdown":
                hotkey = "down"

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = ' '
                    convert_keys.append(key)
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"
        
        elif action_type == "press":
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"
            
            elif hotkey == "arrowup":
                hotkey = "up"
            
            elif hotkey == "arrowdown":
                hotkey = "down"
            
            elif hotkey == "space":
                hotkey = " "
                
            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.press({repr(key_to_press)})"
            
        elif action_type == "keyup":
            key_to_up = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyUp({repr(key_to_up)})"
        
        elif action_type == "keydown":
            key_to_down = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyDown({repr(key_to_down)})"

        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            if content.endswith("\n") or content.endswith("\\n"):
                stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{stripped_content}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"

        
        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                )

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                
                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")
            
            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"
        
        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"
        
        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code


def add_box_token(input_string):
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            
            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(f"{coord_type}='({x},{y})'", f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)
        
        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string




@ray.remote(num_cpus=1)
class EnvWorker():
    system_prompt = uitars_system_prompt
    
    ground_prompt = r"""Output only the coordinate of one point in your response. What element matches the following task: """

    def __init__(self, worker_idx, max_steps=15):
        self.worker_idx = worker_idx
        self.step_timeout = 60
        self.model = 'uitars'
        print('Start to create desktop_env.')
        self.env = DesktopEnv(
            provider_name="docker", 
            action_space="pyautogui",
            screen_size=(1920, 1080),
            cache_dir=f"cache_dirs/cache_{self.worker_idx}",
            headless=True,
            os_type="Ubuntu",
            require_a11y_tree=False
        )

        self.is_init = False
        self.is_done = False
        self.max_steps = max_steps
        # self.parser = Qwen2VLParser()

        self.action_parse_res_factor = 1000
        self.model_type = "qwen25vl"
        self.max_pixels = 16384*28*28
        self.min_pixels = 100*28*28

        self.instruction = None
        self.task_config = None
        self.step_counter = 0
        self.history_images = []
        self.history_messages = []
    
    
    def reset(self, task_config):

        self.instruction = task_config.get("instruction", None)
        self.task_config = task_config
        self.step_counter = 0
        self.is_done = False

        trial_time = 0
        while trial_time < 3:
            try:
                obs = self.env.reset(task_config)
                break
            except Exception as e:
                print(f"Env reset exception: {e}")
                print('Env reset error: ', traceback.format_exc())
                trial_time += 1
        
        if trial_time >= 8:
            self.is_init = True
            self.is_done = True
            print('Env reset failed after 8 trials: ', task_config)
            return {"env_idx": self.worker_idx, "obs_messages": None, "is_done": self.is_done, 'format_reward': 0.0}

        # self.agent.reset()
        self.is_init = True


        init_image = obs["screenshot"]
        # init_image = Image.open(BytesIO(init_image))

        image_base64 = base64.b64encode(BytesIO(init_image).getvalue()).decode("utf-8")

        if self.model == 'steve':
            init_messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_prompt
                        },
                        {
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{image_base64}",
                            "min_pixels": 3136,
                            "max_pixels": 2116800,
                        },
                        {
                            "type": "text",
                            "text": f"User task: {self.instruction}. Please generate the next move."
                        }
                    ]
                }
            ]
        else:
            init_messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Your are a helpful assistant."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_prompt.format(instruction=self.instruction)
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{image_base64}",
                            "min_pixels": 3136,
                            "max_pixels": 2116800,
                        }
                    ]
                }
            ]

        self.history_images = [init_image]
        self.history_messages = init_messages

        return {'env_idx': self.worker_idx, 'obs_messages': self.history_messages, 'is_done': self.is_done, 'format_reward': 0.0}
    
    def step(self, prediction):
        self.is_init = False

        # 1. parse to structure output, might exception
        # 2. parse to pyautogui code, might excpetion
        # 3. env.step, might timeout
        # 4. check the output format(if format is wrong/timeout/exception, return DONE)

        origin_resized_height = obs_image_height = 1080
        origin_resized_width = obs_image_width = 1920

        try:
            parsed_responses = parse_action_to_structure_output(
                prediction,
                self.action_parse_res_factor,
                origin_resized_height,
                origin_resized_width,
                self.model_type,
                self.max_pixels,
                self.min_pixels
            )

            actions = []
            for parsed_response in parsed_responses:
                if "action_type" in parsed_response:
                    if parsed_response["action_type"] == FINISH_WORD:
                        actions = ['DONE']
                        break
                    
                    elif parsed_response["action_type"] == WAIT_WORD:
                        actions = ['WAIT']
                        break
                    
                    elif parsed_response["action_type"] == ENV_FAIL_WORD:
                        actions = ['FAIL']
                        break

                    elif parsed_response["action_type"] == CALL_USER:
                        actions = ['FAIL']
                        break

                pyautogui_code = parsing_response_to_pyautogui_code(
                    parsed_response,
                    obs_image_height,
                    obs_image_width,
                    False # input_swap = False, don't use pyperclip
                )
                actions.append(pyautogui_code)
            
            format_reward = 0.0
        except:
            print('Parse action error: ', prediction)
            print('Error traceback: ', traceback.format_exc())
            format_reward = -1.0
            actions = ['DONE'] # error output format, stop the trajectory immediately

        action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
        for action in actions:
            obs, reward, step_done, info = self.env.step(action)
        
            if step_done:
                self.is_done = True
            
            self.step_counter += 1
            if self.step_counter == self.max_steps:
                self.is_done = True

            step_data = {
                "step_num": self.step_counter,
                "action": action,
                "reward": reward,
                "done": step_done,
                "info": info,
                "action_timestamp": action_timestamp,
            }
        
        self.history_images.append(obs['screenshot'])
        self.history_messages.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": add_box_token(prediction)
            }]
        })

        if not self.is_done:
            if obs['screenshot'] is None:
                self.is_done = True
                # failed to get screenshot
                return {'env_idx': self.worker_idx, 'obs_messages': None, 'is_done': self.is_done, 'format_reward': format_reward}

            image_base64 = base64.b64encode(BytesIO(obs['screenshot']).getvalue()).decode('utf-8')

            self.history_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_base64}",
                        "min_pixels": 3136,
                        "max_pixels": 2116800,
                    }
                ]
            })
            return {'env_idx': self.worker_idx, 'obs_messages': self.history_messages, 'is_done': self.is_done, 'format_reward': format_reward}
        else:
            return {'env_idx': self.worker_idx, 'obs_messages': None, 'is_done': self.is_done, 'format_reward': format_reward}
            
    
    def evaluate(self):
        try:
            return self.env.evaluate()
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0
            
    
    def get_history_messages(self):
        return self.history_messages
    
    def get_history_images(self):
        return self.history_images
    
    def is_done(self):
        return self.is_done

    def is_init(self):
        return self.is_init