# title: Blender AI LLM Addin for Text to 3D Graphics Model
# version: 1.0.0
# date: 2025-1-26
# authors: Taewook Kang
# email: laputa99999@gmail.com
# description: this is a Blender AI LLM Addin for generating Blender Python code using AI models like ChatGPT, Gemma, Llama, etc.
import sys, os, json, argparse, re, textwrap, ast, subprocess, sys, random, math
import bpy, pandas as pd, numpy as np

# Create Blender UI
class OBJECT_PT_CustomPanel(bpy.types.Panel):
	bl_label = "AI Model Selector"
	bl_idname = "OBJECT_PT_custom_panel"
	bl_space_type = 'VIEW_3D'
	bl_region_type = 'UI'
	bl_category = "Gen AI 3D Graphics Model"

	def draw(self, context):
		layout = self.layout
		layout.label(text="Select Model:")
		layout.prop(context.scene, "ai_model", text="")
		layout.label(text="User Prompt:")
		layout.prop(context.scene, "user_prompt", text="")
		layout.operator("object.submit_prompt", text="Submit")

# Blender event operator
class OBJECT_OT_SubmitPrompt(bpy.types.Operator):
	bl_label = "Submit Prompt"
	bl_idname = "object.submit_prompt"

	def execute(self, context):
		option = context.scene.ai_model
		user_prompt = context.scene.user_prompt
		gen_code(option, 'coding Blender python program using bpy, basic grammar without Explanation, "#" inline comments, complicated grammar like lamda and function under user request. do not delete the previous objects. user request is', user_prompt)
		# make box, position (6,-3) with yellow color
		# Create 100 cubes. The y position of each cube follows the cosine function along the x-axis with random color, size.
		# Generate 50 cubes which have positions on each x, y axis on grid style and each cube has random color, size.

		return {'FINISHED'}

# Blender widget registration
def register():
	bpy.utils.register_class(OBJECT_PT_CustomPanel)
	bpy.utils.register_class(OBJECT_OT_SubmitPrompt)

	bpy.types.Scene.ai_model = bpy.props.EnumProperty(
		name="AI Model",
		items=[
				('chatgpt', "ChatGPT", ""),
				('gemma2', "Gemma", ""),
				('llama3.2', "llama", ""),
				('codellama', "codellama", ""),
				('qwen2.5-coder:3b', "qwen2.5", ""),			   
				('vanilj/Phi-4', "Phi", ""),
			]
	)
	bpy.types.Scene.user_prompt = bpy.props.StringProperty(name="User Prompt")

def unregister():
	bpy.utils.unregister_class(OBJECT_PT_CustomPanel)
	bpy.utils.unregister_class(OBJECT_OT_SubmitPrompt)

	del bpy.types.Scene.ai_model
	del bpy.types.Scene.user_prompt

# Import AI, LLM libraries
import openai
from ollama import chat
from ollama import ChatResponse
from openai import OpenAI

# OpenAI model
client = OpenAI(api_key='<input your OpenAI API key>')

def openai_agent(prompt, system_prompt="You are coder for Blender python program", model="gpt-4o"):
	response = client.chat.completions.create(
		model="gpt-4o",
		messages=[
			{
				"role": "system",
				"content": system_prompt
			},
			{
			"role": "user",
			"content": prompt
			}
		],
		temperature=0.1,
		max_tokens=1024,
		top_p=1
	)
	return response.choices[0].message.content

# code agent function
def check_safe_eval(express):
	tokens = express.split()

	try:
		if tokens.index("import") < 0:
			return 
		unsafe_libs = ["os", "shutil", "subprocess", "ctypes", "pickle", "http", "socket", "eval", "exec"]
		unsafe = False
		for lib in unsafe_libs:
			try:
				if tokens.index(lib) >= 0:
					unsafe = True
					break
			except ValueError:
				pass

		if unsafe:
			raise Exception(f"{express} is not safe.")
	except ValueError:
		pass
	return 

def preprocess_code(text: str) -> str:
	try:
		match = re.search(r'```python\n(.*?)```', text, re.DOTALL) # extract code from text between ```python\n and ```
		code = match.group(1).strip()
		code = code.replace('\t', '    ')
		code = textwrap.dedent(code)
		check_safe_eval(code)
		ast.parse(code)
	except IndentationError as e:
		print(f"IndentationError detected: {e}")
		code = ''
	except SyntaxError as e:
		print(f"SyntaxError detected: {e}")
		code = ''
	except Exception as e:
		print(f"Error: {e}")
		code = ''
	return code

def llm_agent(option, prompt):
    response = chat(
        model=option,
        messages=[{"role": "user", "content": prompt}]
    )
    if response and isinstance(response, ChatResponse):
        return response.message.content
    else:
        raise Exception("Failed to get a valid response from the model")

def gen_code(option, instruct_cmd, user_prompt):
	print(f"Selected Option: {option}")
	print(f"User Prompt: {user_prompt}")

	print(f'Calling {option} API...')
	output = ''
	if option == 'chatgpt':
		output = openai_agent(instruct_cmd + ': ' + user_prompt)
	else:
		output = llm_agent(option, instruct_cmd + ': ' + user_prompt)
	print(output)

	for i in range(3):
		try:
			code = preprocess_code(output)
			exec(code)
			print("Code executed successfully.")
			break
		except Exception as e:
			print(f"Error: {e}")
			error_fix_prompt = f"Fix the error {e} in {code}"
			if option == 'chatgpt':
				output = openai_agent(error_fix_prompt)
			else:
				output = llm_agent(option, error_fix_prompt)
			print(output)
			pass

register()
