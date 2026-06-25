import os
import json
import random
import glob
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = r"F:\DataSetF\FaMoS"  # 数据集根目录
OUTPUT_DIR = r"./output_texts_all"  # 文本标注保存目录
RANDOM_SEED = 42  # 【关键】必须与草图生成代码一致，确保划分对齐
# ===========================================

# ------------------------------------------------------------------
# 1. 基于 28 个 FaMoS 动作序列的层级化文本字典
# ------------------------------------------------------------------
# 结构: {动作关键词: [原子级(Atomic), 复合级(Composite), 场景级(Scenario)]}
GPT_EXPANSION_DICT = {
    # --- 基础情绪 (Basic Emotions) ---
    "anger": [
        "The eyebrows squeeze together and lower, while the lips tighten.",
        "A clearly angry expression with a furrowed brow and intense stare.",
        "The person looks furious and aggressive after an unfair situation."
    ],
    "disgust": [
        "The nose wrinkles and the upper lip raises significantly.",
        "A face showing strong disgust and aversion.",
        "The person reacts with repulsion to an unpleasant smell or sight."
    ],
    "fear": [
        "The eyes widen and the eyebrows pull up and inward.",
        "A terrified expression with pale skin and wide eyes.",
        "The person looks scared as if facing an immediate threat."
    ],
    "happiness": [
        "The cheek muscles raise and lip corners pull up diagonally.",
        "A genuine happy face with a bright smile.",
        "The person is beaming with joy after hearing good news."
    ],
    "sadness": [
        "The inner corners of the eyebrows raise and lip corners drop.",
        "A sorrowful expression with a downcast look.",
        "The person looks depressed and is about to cry due to grief."
    ],
    "surprise": [
        "The eyebrows raise high, eyes widen, and the jaw drops.",
        "A shocked face with mouth open in disbelief.",
        "The person is startled by a sudden loud noise."
    ],

    # --- 嘴部特定动作 (Mouth/Lips Specific) ---
    "bareteeth": [
        "The upper and lower lips part to expose the teeth.",
        "Grimacing while showing clenched teeth.",
        "The person is checking their teeth or showing aggression."
    ],
    "blow_cheeks": [
        "The lips close tight and cheeks expand outward with air.",
        "Puffing out both cheeks fully.",
        "The person holds their breath like they are underwater."
    ],
    "cheeks_in": [
        "The cheeks are sucked inwards into the oral cavity.",
        "Making a fish face by sucking in the cheeks.",
        "The person mimics a fish or sucks on a straw strongly."
    ],
    "high_smile": [
        "The lip corners raise sharply towards the eyes, exposing upper teeth.",
        "An ecstatic, high-intensity smile.",
        "The person is laughing uncontrollably at a hilarious joke."
    ],
    "kissing": [
        "The lips protrude forward and pucker into a small circle.",
        "Making a kissing shape with the lips.",
        "The person is blowing a kiss to someone they love."
    ],
    "lip_corners_down": [
        "The triangularis muscle pulls the lip corners downwards.",
        "A distinct frown with a curved mouth.",
        "The person shows disappointment or disapproval."
    ],
    "lips_back": [
        "The lips stretch horizontally towards the ears.",
        "Stretching the mouth wide without opening it.",
        "The person creates a fake polite smile or grimace."
    ],
    "lips_up": [
        "The upper lip elevates to show the upper gum.",
        "Raising the upper lip in a sneer.",
        "The person looks skeptical or mocking."
    ],
    "mouth_down": [
        "The entire mouth structure shifts downwards.",
        "Lowering the mouth with a somber look.",
        "The person is sulking or pouting."
    ],
    "mouth_extreme": [
        "The mouth opens to its maximum possible extent.",
        "An extreme stretching of the jaw and lips.",
        "The person is screaming or yawning widely."
    ],
    "mouth_middle": [
        "The mouth opens slightly to a neutral middle position.",
        "Parting the lips gently.",
        "The person is breathing through their mouth."
    ],
    "mouth_open": [
        "The mandible lowers and the lips part vertically.",
        "Opening the mouth in a standard O-shape.",
        "The person opens their mouth to say 'Ah'."
    ],
    "mouth_side": [
        "The mouth shifts horizontally to one side.",
        "A crooked smirk or side-mouth expression.",
        "The person is chewing on one side or doubting something."
    ],
    "mouth_up": [
        "The entire mouth assembly shifts upwards.",
        "Pushing the mouth up, shortening the philtrum.",
        "The person is making a goofy face."
    ],
    "rolling_lips": [
        "The lips roll inward, disappearing into the mouth.",
        "Biting or hiding the lips completely.",
        "The person is nervous or applying lipstick by rolling lips."
    ],
    "smile_closed": [
        "The lip corners pull up but the lips remain sealed.",
        "A polite smile without showing teeth.",
        "The person is smiling contentedly and calmly."
    ],

    # --- 眉毛与鼻子 (Eyes/Nose) ---
    "eyebrow": [
        "The frontalis muscle contracts to raise both eyebrows.",
        "Raising eyebrows in an arch shape.",
        "The person looks questioning or skeptical."
    ],
    "wrinkle_nose": [
        "The skin on the bridge of the nose crinkles.",
        "Scrunching the nose like a bunny.",
        "The person smells something bad or is sneezing."
    ],

    # --- 下巴与头部运动 (Jaw/Head) ---
    "jaw": [
        "The jaw moves laterally from side to side.",
        "Grinding or moving the jaw sideways.",
        "The person is adjusting their jaw or grinding teeth."
    ],
    "head_rotation_left_right": [
        "The head rotates on the yaw axis from left to right.",
        "Shaking the head 'no' horizontally.",
        "The person is looking around the room or denying something."
    ],
    "head_rotation_up_down": [
        "The head rotates on the pitch axis, looking up and down.",
        "Nodding the head 'yes' vertically.",
        "The person is looking at the sky and then the floor."
    ],

    # --- 说话序列 (Speech) ---
    "sentence": [
        "The lips and jaw move rhythmically in complex patterns.",
        "The person is speaking a full sentence with natural articulation.",
        "The person is having a conversation or giving a speech."
    ]
}

# 默认回退描述（以防万一有未覆盖的词）
DEFAULT_TEMPLATES = [
    "Facial movement characterized by {action}.",
    "The person is performing the {action} motion.",
    "A 3D facial animation showing {action}."
]


def find_matching_key(folder_name):
    """
    精确匹配逻辑：检查文件夹名是否包含字典中的关键词
    folder_name 示例: "Seq_anger", "FaMoS_sentence_01"
    """
    clean_name = folder_name.lower()

    # 优先匹配长词，防止 "mouth_open" 被匹配成 "mouth" (如果字典里有mouth的话)
    # 我们按照字典key的长度降序排列来匹配
    sorted_keys = sorted(GPT_EXPANSION_DICT.keys(), key=len, reverse=True)

    for key in sorted_keys:
        # 如果关键词完整出现在文件夹名中
        if key in clean_name:
            return key

    return None


def process_subset(subjects, split_name):
    """处理特定的数据集子集并生成 JSON"""
    dataset_list = []

    print(f"正在处理 {split_name} 集 ({len(subjects)} 人)...")

    missing_matches = 0

    for subj in tqdm(subjects):
        subj_path = os.path.join(DATA_ROOT, subj)
        if not os.path.exists(subj_path): continue

        # 获取该受试者的所有动作序列文件夹
        action_folders = sorted([d for d in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, d))])

        for action_folder in action_folders:
            # 1. 匹配动作关键词
            key = find_matching_key(action_folder)

            if key:
                descriptions = GPT_EXPANSION_DICT[key]
            else:
                # 如果没匹配到，使用通用模板
                # print(f"Warning: No match found for {action_folder}")
                clean_name = action_folder.replace("Seq_", "").replace("_", " ")
                descriptions = [t.format(action=clean_name) for t in DEFAULT_TEMPLATES]
                missing_matches += 1

            # 2. 构造数据条目
            # 注意：这里的 sketch_filename 必须与 generate_sketches.py 生成的文件名完全一致
            # 生成规则: f"{subj}_{seq_name}_sketch.png"
            entry = {
                "subject_id": subj,
                "action_id": action_folder,
                "sketch_filename": f"{subj}_{action_folder}_sketch.png",  # 对应图片文件名
                "mesh_seq_path": os.path.join(subj, action_folder).replace("\\", "/"),  # 统一用正斜杠
                "text_captions": descriptions  # 包含 3 个层级的描述
            }
            dataset_list.append(entry)

    # 保存为 JSON
    save_path = os.path.join(OUTPUT_DIR, f"text_annotations_{split_name}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_list, f, indent=4)

    print(f"  -> 已保存 {len(dataset_list)} 条标注至 {save_path}")
    if missing_matches > 0:
        print(f"  -> 警告: 有 {missing_matches} 个文件夹未匹配到预定义字典，使用了通用模板。")


def main():
    if not os.path.exists(DATA_ROOT):
        print(f"错误：数据集路径不存在 {DATA_ROOT}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 获取受试者列表 (FaMoS_subject_xxx)
    subjects = sorted([d for d in os.listdir(DATA_ROOT) if d.startswith("FaMoS_subject")])
    print(f"总受试者人数: {len(subjects)}")

    # 2. 【关键】使用与草图生成完全一致的随机种子和逻辑
    # 必须保证这里的 train/val/test 划分和草图生成时的一模一样
    random.seed(RANDOM_SEED)
    random.shuffle(subjects)

    n_train = int(len(subjects) * 0.8)
    n_val = int(len(subjects) * 0.1)

    train_subs = subjects[:n_train]
    val_subs = subjects[n_train: n_train + n_val]
    test_subs = subjects[n_train + n_val:]

    print(f"数据集划分 (Seed={RANDOM_SEED}):")
    print(f"  Train: {len(train_subs)} | Val: {len(val_subs)} | Test: {len(test_subs)}")

    # 3. 生成标注文件
    process_subset(train_subs, "train")
    process_subset(val_subs, "val")
    process_subset(test_subs, "test")

    print("\n所有文本标注生成完毕！")
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()