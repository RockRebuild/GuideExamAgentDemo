import os
os.environ['FLAGS_use_onednn'] = '0'   # 必须放在 import paddleocr 之前
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'    # 同样禁用 MKLDNN
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 确保不用 GPU
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='ch')   # 中英混合用 lang='ch'
result = ocr.predict('全国导游基础知识统编教材.pdf')

# 按页收集文本
pages_text = []
for page in result:
    if page is None:   # 空白页可能为 None
        pages_text.append("")
        continue
    lines = [line[1][0] for line in page]   # 取每行的文字部分
    page_text = "\n".join(lines)
    pages_text.append(page_text)

# 打印前3页预览
for i, text in enumerate(pages_text[:3]):
    print(f"=== 第{i+1}页 ===\n{text[:200]}\n")
