import json
import argparse

def _response_to_traj(resp: str):
    """将一个字符串形式的轨迹解析为 list[list[float]]。

    期望格式类似："[x,y,h], [x,y,h], ..."，中间允许有空格和换行，
    也允许简单的标签包裹（例如含有 "<answer>" 字样，会直接删除这些标记）。
    解析失败时返回 None。
    """
    if not isinstance(resp, str):
        return None

    # 去掉首尾空白和换行
    s = resp.strip().replace("\n", " ")
    if s.startswith(">["):
        s = s[1:]

    # 粗暴去掉简单标签标记（不做复杂解析）
    for tag in ["<answer>", "</answer>", "<|im_end|>","<|start-latent|>","<|latent|>","<|end-latent|>","<|start-latent-vis|>","<|end-latent-vis|>","<|latent-vis|>","<|im_end|>", "\n"]:
        s = s.replace(tag, "")
    try:
        # 补上最外层中括号，使其成为合法 JSON 数组
        try:
            arr = json.loads("[" + s + "]")
            # print(arr)
            # 转成 list[list[float]]
            return [[float(v) for v in point] for point in arr]
        except Exception as e:
            ## prefilling add a extra '['
            arr = json.loads("[[" + s + "]")
            # print(arr)
            # 转成 list[list[float]]
            return [[float(v) for v in point] for point in arr]
    except Exception as e:
        print(s)
        print(e)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--ref_path", type=str, default="output/navsim/navsim_results_eval.json")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--is_cot", action="store_true")
    args = parser.parse_args()

    with open(args.ref_path, 'r') as f:
        ref_data = json.load(f)

    with open(args.input_path, 'r') as f:
        input_data = json.load(f)

    img_curid_map = {}
    for item in ref_data["predictions"]:
        img_path = item["messages"][0]["content"][-2]["image"].replace("file://", "")
        img_curid_map[img_path] = item["id"]

    img_pred_map = {}
    for item in input_data:
        if args.is_cot:
            answer = item["output_text"].split("</think>")[1]
            pred = _response_to_traj(answer)
        else:
            pred = _response_to_traj(item["output_text"])
        img = item["messages"][0]["content"][0]["image"]
        img_pred_map[img] = pred

    ## replace the pred in ref_data with the pred in input_data
    for item in ref_data["predictions"]:
        img_path = item["messages"][0]["content"][-2]["image"].replace("file://", "")
        item["pre_traj"] = img_pred_map[img_path]


    with open(args.output_path, 'w') as f:
        json.dump(ref_data, f, indent=4)

    
   