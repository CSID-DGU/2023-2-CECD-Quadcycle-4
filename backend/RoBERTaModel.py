from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# tokenizer 불러오기
tokenizer_name = "roberta-base"
trained_tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)

# 모델 불러오기
model_name = "./roberta_model"
model = RobertaForSequenceClassification.from_pretrained(model_name)

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
    result = torch.argmax(logits, dim=1).item()

    return classes[result]