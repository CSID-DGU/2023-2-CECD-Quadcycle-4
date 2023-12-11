from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

model_dir = "./roberta_model"

# tokenizer 불러오기
trained_tokenizer = RobertaTokenizer.from_pretrained(model_dir)
# 모델 불러오기
model = RobertaForSequenceClassification.from_pretrained(model_dir)

# 예측 클래스 리스트
classes = ['SUM','SUMIF','ROUND','ROUNDDOWN','ROUNDUP','INT','ABS','SQRT','EXP','FACT',
           'PI','MOD','PRODUCT','SUMPRODUCT','POWER', 'TRUNC',
           'AVERAGE','MAX','MIN','RANK','LARGE','SMALL','COUNT','COUNTA','COUNTBLANK','COUNTIF',
           'DSUM','DAVERAGE','DMAX','DMIN','DCOUNT','DGET','DPRODUCT','DSTDEV','DVAR','ISERROR',
           'FIND','SEARCH','MID','LEFT','RIGHT','LOWER','UPPER','PROPER','TRIM','LEN',
           'REPLACE','CONCATENATE','REPT','VALUE',
           'VLOOKUP','HLOOKUP','CHOOSE','INDEX','MATCH','OFFSET']


def find_classification(text):

    inputs = trained_tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)

    logits = output.logits
    predicted = torch.argmax(logits, dim=1).item()
    probabilites = torch.softmax(logits, dim=1).tolist()[0]

    # 추론 확률이 50% 미만이면 Invalid
    if probabilites[predicted] < 0.5:
        return "Invalid"

    return classes[predicted]