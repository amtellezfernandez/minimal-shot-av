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
    parser.add_argument("--candidate_index", type=int, default=0,
                        help="Candidate index to export when the input JSON "
                             "contains a candidates list")
    parser.add_argument("--all_candidates_dir", type=str, default=None,
                        help="If set, export one NAVSIM eval JSON per "
                             "candidate index into this directory")
    args = parser.parse_args()

    with open(args.ref_path, 'r') as f:
        ref_data = json.load(f)

    with open(args.input_path, 'r') as f:
        input_data = json.load(f)

    img_curid_map = {}
    for item in ref_data["predictions"]:
        img_path = item["messages"][0]["content"][-2]["image"].replace("file://", "")
        img_curid_map[img_path] = item["id"]

    def candidate_pred(item, candidate_index):
        if "candidates" in item:
            candidates = item["candidates"]
            if candidate_index >= len(candidates):
                return None
            candidate = candidates[candidate_index]
            traj = candidate.get("trajectory")
            if traj is not None:
                return traj
            return _response_to_traj(candidate.get("raw_text", ""))
        if args.is_cot:
            answer = item["output_text"].split("</think>")[1]
            return _response_to_traj(answer)
        return _response_to_traj(item["output_text"])

    def build_prediction_map(candidate_index):
        img_pred_map = {}
        for item in input_data:
            pred = candidate_pred(item, candidate_index)
            img = item["messages"][0]["content"][0]["image"]
            img_pred_map[img] = pred
        return img_pred_map

    def apply_predictions(prediction_map):
        output = json.loads(json.dumps(ref_data))
        for item in output["predictions"]:
            img_path = item["messages"][0]["content"][-2]["image"].replace("file://", "")
            item["pre_traj"] = prediction_map[img_path]
        return output

    if args.all_candidates_dir:
        candidate_count = max(len(item.get("candidates", [])) for item in input_data)
        for candidate_index in range(candidate_count):
            output = apply_predictions(build_prediction_map(candidate_index))
            out_path = f"{args.all_candidates_dir.rstrip('/')}/candidate_{candidate_index}.json"
            with open(out_path, 'w') as f:
                json.dump(output, f, indent=4)
    else:
        output = apply_predictions(build_prediction_map(args.candidate_index))
        with open(args.output_path, 'w') as f:
            json.dump(output, f, indent=4)

    
   
