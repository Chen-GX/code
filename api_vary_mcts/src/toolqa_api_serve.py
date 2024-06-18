import os, sys
os.chdir(sys.path[0])
import argparse

from tool_online import ToolQA_OnLine

from flask import Flask, request, jsonify

app = Flask(__name__)

def create_tool_agent(tool_device, path):
    args = argparse.Namespace(tool_device=tool_device, path=path)
    return ToolQA_OnLine(args)

tool_agent = create_tool_agent(tool_device=0, path="/mnt/workspace/nas/chenguoxin.cgx/api/datasets/ToolQA")
# tool_agent = create_tool_agent(tool_device=0, path="/yinxr/workhome/zzhong/chenguoxin/api/datasets/ToolQA")

@app.route('/toolqa', methods=['POST'])
def call_toolqa():
    if request.is_json:
        data = request.json
        new_action_type = data.get('new_action_type', '')
        new_params = data.get('new_params', "{}")

        observation = tool_agent.parse_and_perform_action(new_action_type, new_params)

        result = {"observation": observation}
        return jsonify(result)
    else:
        return jsonify({"error": "Request must be JSON"}), 400

# if __name__ == '__main__':
#     # 在开发环境中运行
#     # 注意：Flask自带的服务器适用于开发但不推荐在生产环境中使用。
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--tool_device", type=int, default=1)
#     parser.add_argument("--path", type=str, default="/yinxr/workhome/zzhong/chenguoxin/datasets/ToolQA")
#     args = parser.parse_args()
    # app.run(host='0.0.0.0', port=5010, debug=False)
