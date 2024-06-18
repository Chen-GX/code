import requests

def test_toolqa(url, new_action_type, new_params):
    data = {
        'new_action_type': new_action_type,
        'new_params': new_params,
    }
    print(data)
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()  # 返回响应的JSON数据
    else:
        return response.status_code, response.text

if __name__ == '__main__':
    # 把URL替换成你的Flask服务器地址
    url = 'http://127.0.0.1:5010/toolqa'
    
    # new_action_type = 'RetrieveAgenda'
    # new_params = {"keyword": "Stephen's Opera performance"}

    new_action_type = 'RetrieveScirex'
    new_params = {"keyword": "Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task"}

    # 调用test_toolqa并打印返回结果
    data = test_toolqa(url, new_action_type, new_params)
    print(data)
