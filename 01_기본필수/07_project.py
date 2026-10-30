# from pathlib import Path
# import pandas as pd
# import torch
# from torch import nn

# torch.set_num_threads(1)
# base = Path(__file__).resolve().parent
# import matplotlib
# matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
# import matplotlib.pyplot as plt
# df = pd.read_csv(base / "points.csv")
# train = df[df["split"] == "train"]
# valid = df[df["split"] == "valid"]
# X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
# y_train = torch.tensor(train["label"].values, dtype=torch.long)
# X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32)
# y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# # test 4건은 읽은 표에만 있으며 학습/평가 Tensor로 만들지 않습니다.

# # 실습 07~08: 기준 lr과 비교 lr만 다릅니다. 더 나쁘게 나와도 그대로 보고합니다.
# loss_fn = nn.CrossEntropyLoss()
# summary = []
# for name, lr in [("baseline", 0.001), ("candidate", 0.01)]:
#     torch.manual_seed(42)
#     model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
#     history = []
#     for epoch in range(1, 21):
#         model.train()
#         optimizer.zero_grad()
#         loss = loss_fn(model(X_train), y_train)
#         # 이미 zero_grad와 loss 계산이 끝났습니다. loss.backward() 다음 optimizer.step()을 호출하세요.
#         # 학습 데이터만 사용합니다. 비교 후보마다 모델을 처음부터 새로 만듭니다.
#         # >>> B08
#         # TODO B08: 위 요구사항에 맞게 이 블록만 완성합니다.
#         pass
#         # <<< B08
#         model.eval()
#         with torch.no_grad():
#             train_loss = loss_fn(model(X_train), y_train).item()
#             valid_loss = loss_fn(model(X_valid), y_valid).item()
#         # 한 epoch의 번호와 갱신 후 train_loss, valid_loss를 기록하세요.
#         # history.append([epoch, train_loss, valid_loss]) 형태로 추가합니다.
#         # >>> B09
#         # TODO B09: 위 요구사항에 맞게 이 블록만 완성합니다.
#         history.append([epoch, train_loss, 0.0])
#         # <<< B09
#     table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss"])
#     table.to_csv(base / f"history_{name}.csv", index=False)
#     best = table.loc[table["valid_loss"].idxmin()]
#     summary.append([name, lr, int(best["epoch"]), best["valid_loss"]])
#     plt.figure(figsize=(8, 5))
#     plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval mode)")
#     plt.plot(table["epoch"], table["valid_loss"], label="valid")
#     plt.axvline(best["epoch"], linestyle="--", label="best epoch")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(base / f"curves_{name}.png", dpi=150)
#     plt.close()
# result = pd.DataFrame(summary, columns=["setting", "lr", "best_epoch", "best_valid_loss"])
# result.to_csv(base / "experiment_table.csv", index=False)
# print(result.round(6).to_string(index=False))
# print("test 미사용 / best epoch는 관찰한 최소 지점, 기본필수에서는 모델 저장·복원하지 않음")




from pathlib import Path
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)  # 인프라 제어: CPU 연산 스레드를 1개로 제한하여, 배포 파이프라인 서버 부하 상태와 관계없이 항상 재현 가능한 일정한 메트릭을 확보합니다.
base = Path(__file__).resolve().parent  # 경로 확보: 사내 개발 서버 내 본 소스코드 파일의 물리적인 절대 위치를 자동 인식합니다.
import matplotlib
matplotlib.use("Agg")  # 배포 자동화: 리눅스 도커(Docker) 컨테이너처럼 모니터 화면이 없는 CLI 서버에서도 에러 없이 그래프 이미지(.png)를 내보내도록 설정합니다.
import matplotlib.pyplot as plt

df = pd.read_csv(base / "points.csv")  # 데이터 적재: 데이터 마트에서 정제해 둔 과거 업무 처리 이력 데이터인 'points.csv'를 시스템으로 읽어옵니다.
train = df[df["split"] == "train"]  # 기출 데이터: 모델이 비즈니스 패턴(가중치)을 직접 학습할 과거 업무 자료 묶음
valid = df[df["split"] == "valid"]  # 모의고사 데이터: 모델이 본질적인 원리를 깨우쳤는지 블라인드 테스트할 별도의 검증 자료

# 데이터 전처리: 판다스 표 데이터를 PyTorch 파이프라인 연산이 가능한 Tensor(텐서) 형태로 변환합니다.
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# test 4건은 읽은 표에만 있으며 학습/평가 Tensor로 만들지 않습니다.

# 실습 07~08: 기준 lr과 비교 lr만 다릅니다. 더 나쁘게 나와도 그대로 보고합니다.
loss_fn = nn.CrossEntropyLoss()  # 지표 정의: 모델의 예측값과 실제 업무 처리 정답 간의 괴리율을 채점하여 오차 점수를 내는 '벌점 채점관(손실 함수)'입니다.
summary = []  # 파이널 보고서 작성: 두 실험군의 최고 성능 시점과 메트릭 정보를 요약하여 모아둘 상자입니다.

for name, lr in [("baseline", 0.001), ("candidate_1", 0.01), ("candidate_2", 0.1), ("candidate_3", 0.2)]:
    # candidate_2, candidate_3 : 2개를 추가하여 lr를 0.1와 0.2 변경하여 진행 
    torch.manual_seed(42)  # 통제 변수 고정: 두 실험 모델의 초기 시작 파라미터 상태를 완벽하게 통일하여 불공정한 비교를 원천 차단합니다.
    
    # 아키텍처 빌드: 2차원 업무 특징을 받아 복잡한 비선형 경계선을 그릴 수 있는 미니 ANN 구조를 조립합니다.
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    
    # 해결사 선언: 현업 수렴 속도가 가장 뛰어난 AdamW를 가중치 조정자로 세팅합니다. (weight_decay=0 고정)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history = []  # 모니터링 대장: 이번 실험의 진행 성적을 에폭별로 기록해 둘 원장입니다.
    
    for epoch in range(1, 21):  # 20바퀴(에폭) 동안의 학습 레이스를 시작합니다.
        model.train()  # 단계 1: 모델을 '학습 모드'로 선언합니다.
        optimizer.zero_grad()  # 단계 2: 메모리에 남아있던 이전 미니배치의 기울기 연산 잔여 흔적을 깨끗이 청소합니다.
        loss = loss_fn(model(X_train), y_train)  # 단계 3: 전체 학습 데이터를 풀게 한 뒤 채점 벌점을 받습니다.
        
        # 이미 zero_grad와 loss 계산이 끝났습니다. loss.backward() 다음 optimizer.step()을 호출하세요.
        # 학습 데이터만 사용합니다. 비교 후보마다 모델을 처음부터 새로 만듭니다.
        # >>> B08
        loss.backward()  # 단계 4: 역전파를 수행하여 벌점을 낮추기 위한 가중치 수정 방향(기울기)을 자동으로 계산합니다.
        optimizer.step()  # 단계 5: 계산된 미분값과 해결사 규칙에 따라 실제 파라미터(신경망 가중치)를 한 걸음 업데이트합니다.
        # <<< B08
        
        # 에폭 말 실력 정밀 진단: 1바퀴의 학습이 끝나는 즉시 무작위 요소를 배제하고 순수한 학습 성과를 도출합니다.
        model.eval()  # 단계 6: 모델을 '평가 모드'로 스위칭합니다. 실전 서비스 환경과 똑같이 Dropout 등을 끕니다.
        with torch.no_grad():  # 단계 7: 시험 중에는 가중치를 고치지 않으므로 미분 그래프 추적을 차단하여 속도와 서버 메모리를 절약합니다.
            train_loss = loss_fn(model(X_train), y_train).item()  # 공부했던 기출문제 복습 벌점 점수
            valid_loss = loss_fn(model(X_valid), y_valid).item()  # 낯선 시험 문제(검증 데이터)에 대한 실전 오차 벌점 점수
            
        # 한 epoch의 번호와 갱신 후 train_loss, valid_loss를 기록하세요.
        # history.append([epoch, train_loss, valid_loss]) 형태로 추가합니다.
        # >>> B09
        history.append([epoch, train_loss, valid_loss])  # 이번 바퀴의 메트릭 성적을 대장에 올바르게 누적 적재합니다.
        # <<< B09
        
    # 성적표 가공 및 개별 실험 저장 처리
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss"])
    table.to_csv(base / f"history_{name}.csv", index=False)  # 에폭별 원시 로그를 CSV 아카이브로 저장합니다.
    
    best = table.loc[table["valid_loss"].idxmin()]  # 모의고사 벌점(valid_loss)이 가장 낮았던 '최고 전성기 에폭'을 검출합니다.
    summary.append([name, lr, int(best["epoch"]), best["valid_loss"]])  # 전체 비교 테이블에 요약 정보를 누적합니다.
    
    # 시각화 보고서: 오버피팅 발생 징후와 학습률별 억제력을 한눈에 볼 수 있도록 벌점 곡선 그래프를 렌더링합니다.
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval mode)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")  # 최적의 배포 골든 타이밍 위치에 수직 가이드 점선을 그립니다.
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base / f"curves_{name}.png", dpi=150)  # 그래프를 아티팩트 이미지 파일로 출력합니다.
    plt.close()

# 최종 의사결정: 경영진 보고 및 프로덕션 배포 공유를 위해 통합 요약 성적표를 파일로 완성 내보내고 화면에 출력합니다.
result = pd.DataFrame(summary, columns=["setting", "lr", "best_epoch", "best_valid_loss"])
result.to_csv(base / "experiment_table.csv", index=False)
print(result.round(6).to_string(index=False))
print("test 미사용 / best epoch는 관찰한 최소 지점, 기본필수에서는 모델 저장·복원하지 않음")
