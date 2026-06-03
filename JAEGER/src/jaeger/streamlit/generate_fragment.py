import os
import argparse
from mol_tree import MolTree

# 原有 import 保持
# from your_project import run_jaeger_generation   # 这里替换为你原来调用的生成函数


def generate_with_fragment(seed_smiles, target_fragment, num_samples=10):
    """
    基于 seed 分子，只替换指定 fragment 生成新分子
    """
    tree = MolTree(seed_smiles)
    # 找到包含目标 fragment 的节点
    target_nodes = [n for n in tree.nodes if target_fragment in n.smiles]

    new_smiles_list = []
    for node in target_nodes:
        node.assemble()
        # 取候选片段进行替换
        for cand in node.cands[:num_samples]:
            new_smiles_list.append(cand)

    return list(set(new_smiles_list))


def main(args):
    if args.fragment:
        print(f"🔎 Using fragment-specific generation with fragment {args.fragment}")
        # 用输入 SMILES 做基准（比如从种子文件读入）
        with open(args.seed_file) as f:
            seed_smiles_list = [line.strip() for line in f if line.strip()]

        all_results = []
        for smi in seed_smiles_list:
            results = generate_with_fragment(smi, args.fragment, num_samples=args.nsamples)
            all_results.extend(results)

        os.makedirs(args.save_dir, exist_ok=True)
        out_file = os.path.join(args.save_dir, f"fragment_{args.fragment.replace('/', '_')}.smi")
        with open(out_file, "w") as f:
            for smi in all_results:
                f.write(smi + "\n")
        print(f"✅ Fragment-specific molecules saved to {out_file}")

    else:
        print("🚀 Running standard Jaeger latent space generation...")
        # 调用原有的 latent space 生成逻辑
        # run_jaeger_generation(args)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 原有参数
    parser.add_argument("--save_dir", type=str, default="results", help="保存结果的目录")
    parser.add_argument("--seed_file", type=str, default="seeds.smi", help="输入种子 SMILES 文件")
    parser.add_argument("--nsamples", type=int, default=10, help="每个 fragment 替换的样本数")

    # 新增参数：fragment-specific generation
    parser.add_argument(
        "--fragment",
        type=str,
        default=None,
        help="Fragment SMILES to constrain generation (fragment-specific molecule generation)"
    )

    args = parser.parse_args()
    main(args)
