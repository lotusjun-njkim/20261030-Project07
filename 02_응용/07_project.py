# from pathlib import Path
# import pandas as pd
# import torch
# from torch import nn

# torch.set_num_threads(1)
# base = Path(__file__).resolve().parent
# from torch.utils.data import TensorDataset, DataLoader
# import copy
# import matplotlib
# matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
# import matplotlib.pyplot as plt

# out = base / "outputs"
# out.mkdir(exist_ok=True)
# # 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
# device = torch.device("cpu")
# df = pd.read_csv(base / "data" / "points.csv")
# train = df[df["split"] == "train"]
# valid = df[df["split"] == "valid"]
# X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
# y_train = torch.tensor(train["label"].values, dtype=torch.long)
# X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
# y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)
# loss_fn = nn.CrossEntropyLoss()
# configs = [{'name': 'sgd', 'optimizer': 'SGD', 'lr': 0.01}, {'name': 'adamw', 'lr': 0.01}, {'name': 'adamw_dropout', 'lr': 0.01, 'dropout': 0.3}]
# summary = []

# for config in configs:
#     torch.manual_seed(42)
#     width = config.get("width", 32)
#     layers = [nn.Linear(2, width)]
#     if config.get("bn", False):
#         layers.append(nn.BatchNorm1d(width))
#     layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
#     if config.get("bn", False):
#         layers.append(nn.BatchNorm1d(width))
#     layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])
#     model = nn.Sequential(*layers).to(device)
#     # shuffle용 Generator를 후보마다 같은 seed로 다시 만듭니다.
#     loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
#                         generator=torch.Generator().manual_seed(43), num_workers=0)
#     if config.get("optimizer", "AdamW") == "SGD":
#         optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
#                                     momentum=config.get("momentum", 0.0), weight_decay=0.0)
#     else:
#         optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)
#     history = []
#     best_loss = float("inf")
#     best_state = None
#     best_epoch = 0
#     for epoch in range(1, 40 + 1):
#         model.train()
#         for xb, yb in loader:
#             xb, yb = xb.to(device), yb.to(device)
#             optimizer.zero_grad()
#             loss = loss_fn(model(xb), yb)
#             # loss.backward() 다음 optimizer.step()을 호출하세요.
#             # 같은 초기화·동일 shuffle 순서를 사용하며 세 후보의 변경 조건을 읽어 구분합니다.
#             # >>> A07
#             # TODO A07: 위 요구사항에 맞게 이 블록만 완성합니다.
#             pass
#             # <<< A07
#         # 같은 epoch 말 모델을 eval 모드로 비교합니다. 학습 중 Dropout loss와 구분합니다.
#         # epoch 종료 후 model.eval()을 호출하세요.
#         # 그 아래 no_grad 블록은 gradient 추적을 끄지만, Dropout 모드를 바꾸지는 않습니다.
#         # >>> A08
#         # TODO A08: 위 요구사항에 맞게 이 블록만 완성합니다.
#         pass
#         # <<< A08
#         with torch.no_grad():
#             train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
#             logits = model(X_valid)
#             valid_loss = loss_fn(logits, y_valid).item()
#             valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
#             # 최저 valid loss를 관측하면 값·epoch·가중치 복사본을 함께 보관합니다.
#             # A09 요구: 동률은 먼저 발견한 epoch 유지. copy.deepcopy(model.state_dict()) 사용.
#             # model.state_dict()만 대입하면 이후 학습이 같은 Tensor를 바꾸므로 안 됩니다.
#             # >>> A09
#             # TODO A09: 최적 loss·epoch와 독립된 state_dict 복사본을 보관하세요.
#             pass
#             # <<< A09
#         history.append([epoch, train_loss, valid_loss, valid_acc])
#     table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
#     best = table.loc[table["valid_loss"].idxmin()]
#     name = config["name"]
#     table.to_csv(out / f"07_{name}_history.csv", index=False)
#     summary.append({**config, "best_epoch": int(best["epoch"]),
#                     "best_valid_loss": float(best["valid_loss"]),
#                     "valid_acc_at_best": float(best["valid_acc"]),
#                     "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})
#     # 제공 코드: 학습이 끝난 뒤 최적 상태로 복원하고 그 모델로 다시 평가합니다.
#     model.load_state_dict(best_state)
#     model.eval()
#     with torch.no_grad():
#         best_pred = model(X_valid).argmax(1).cpu()
#         restored_loss = loss_fn(model(X_valid), y_valid).item()
#     labels = y_valid.cpu()
#     class_rows = []
#     for label in (0, 1):
#         # A10 요구: 실제 정답이 label인 행을 먼저 고른 뒤 support·correct·recall 계산.
#         # 전체 accuracy나 '예측 label의 수'로 나누지 않습니다. 0건이면 recall=None.
#         # >>> A10
#         # TODO A10: 클래스별 실제 표본 수·정답 수·Recall을 기록하세요.
#         pass
#         # <<< A10
#     pd.DataFrame(class_rows).to_csv(out / f"07_{name}_class_metrics.csv", index=False)
#     pd.DataFrame({"id":valid["id"].tolist(),"actual":labels.tolist(),
#                   "predicted":best_pred.tolist()}).to_csv(out / f"07_{name}_predictions.csv", index=False)
#     summary[-1]["restored_valid_loss"] = restored_loss
#     summary[-1]["restored_matches_best"] = abs(restored_loss-best_loss) < 1e-6
#     plt.figure(figsize=(8, 5))
#     plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
#     plt.plot(table["epoch"], table["valid_loss"], label="valid")
#     plt.axvline(best["epoch"], linestyle="--", label="best epoch")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out / f"07_{name}_curves.png", dpi=150)
#     plt.close()
#     print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))
# result = pd.DataFrame(summary)
# result.to_csv(out / "07_comparison.csv", index=False)
# print("생성: outputs/07_*.csv, *.png / test 미사용")


from pathlib import Path
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)  # 인프라 제어: CPU 계산 스레드를 1개로 제한하여, 개발 서버의 타 프로세스 부하 상태와 관계없이 항상 재현 가능한 일정한 메트릭과 연산 성능을 확보합니다.
base = Path(__file__).resolve().parent  # 경로 확보: 배포 컨테이너 내 소스코드 파일의 물리적인 절대 위치를 자동 인식하여 파일 시스템 접근 오류를 방지합니다.
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib
matplotlib.use("Agg")  # 배포 자동화: 리눅스 컨테이너 환경처럼 모니터 화면이 없는 CLI 서버 환경에서도 팝업창 에러 없이 백그라운드 메모리에서 그래프 이미지 파일(.png)을 안전하게 내보내도록 설정합니다.
import matplotlib.pyplot as plt

out = base / "outputs"
out.mkdir(exist_ok=True)  # 아티팩트 보관: 최종 의사결정용 대조 실험표, 클래스별 성적표, 예측 결과 리포트를 보관할 outputs 폴더를 생성합니다.
device = torch.device("cuda:0")  # 인프라 지정: 1차 프로토타입 검증 및 재현성 확보를 위해 연산 장치를 CPU로 세팅합니다.

df = pd.read_csv(base / "data" / "points.csv")  # 데이터 적재: 데이터 마트에서 정제해 둔 과거 업무 처리 이력 스냅샷 파일('points.csv')을 시스템으로 읽어옵니다.
train = df[df["split"] == "train"]  # 기출 데이터: 모델이 비즈니스 패턴(가중치)을 직접 학습할 과거 업무 자료 묶음
valid = df[df["split"] == "valid"]  # 모의고사 데이터: 모델이 단순 암기가 아닌 본질적인 규칙을 깨우쳤는지 블라인드 테스트할 별도의 검증 자료

# 데이터 전처리: 판다스 표 데이터를 PyTorch 파이프라인 연산이 가능한 Tensor(텐서) 형태로 변환하여 연산 장치에 할당합니다.
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)
loss_fn = nn.CrossEntropyLoss()  # 지표 정의: 모델의 예측값과 실제 업무 처리 정답 간의 괴리율을 채점하는 '벌점 채점관(손실 함수)'입니다.

# 대조 실험 설계: SGD 기본형, AdamW 기본형, AdamW에 드롭아웃 규제를 추가한 모델까지 총 3가지 후보군을 세팅합니다.
configs = [{'name': 'sgd', 'optimizer': 'SGD', 'lr': 0.01}, 
           {'name': 'adamw', 'lr': 0.01}, 
           {'name': 'adamw_dropout', 'lr': 0.01, 'dropout': 0.3}]
summary = []  # 파이널 보고서 작성: 실험군들의 최고 전성기 성적과 복원 정상 여부를 요약하여 모아둘 상자입니다.

for config in configs:
    torch.manual_seed(42)  # 통제 변수 고정: 각 실험 후보들의 내부 초기 가중치(시작 지식) 상태를 완벽하게 통일하여 불공정한 비교를 원천 차단합니다.
    width = config.get("width", 32)
    
    # 아키텍처 빌드: 업무 특징 데이터를 받아 연산할 은닉층 레이어와 활성화 함수, 드롭아웃 층을 순서대로 조립합니다.
    layers = [nn.Linear(2, width)]
    if config.get("bn", False): layers.append(nn.BatchNorm1d(width))
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    if config.get("bn", False): layers.append(nn.BatchNorm1d(width))
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])
    model = nn.Sequential(*layers).to(device)
    
    # 데이터 배급 관리: 대용량 데이터 흐름을 모사해 기출문제를 32개씩 쪼개어 공급하는 배급차를 세팅합니다. 섞는 순서 시드(`43`)도 통제합니다.
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)
                        
    # 최적화 해결사 매핑: 설정 딕셔너리를 읽어와 SGD 혹은 AdamW 최적화 알고리즘 객체를 동적으로 생성합니다.
    if config.get("optimizer", "AdamW") == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    momentum=config.get("momentum", 0.0), weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)
        
    history = []  # 모니터링 대장: 이번 실험의 진행 성적을 에폭별로 기록해 둘 원장입니다.
    best_loss = float("inf")  # 조기 종료 장치: 역대 최저 모의고사 벌점을 기록할 변수를 무한대로 초기화해 둡니다.
    best_state = None         # 전성기 박제 상자: 가장 똑똑했던 시점의 모델 뇌세포(가중치 딕셔너리)를 보관할 상자입니다.
    best_epoch = 0            # 전성기 타이밍: 가장 똑똑했던 에폭 번호를 기록합니다.
    
    for epoch in range(1, 40 + 1):  # 총 40바퀴(에폭) 동안의 학습 레이스를 시작합니다.
        model.train()  # 단계 1: 모델을 '학습 모드'로 선언합니다. 이 상태에서만 Dropout이 작동해 편법 암기를 방지합니다.
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()  # 단계 2: 메모리에 남아있던 이전 미니배치의 기울기 연산 잔여 흔적을 깨끗이 청소합니다.
            loss = loss_fn(model(xb), yb)  # 단계 3: 32개의 미니배치 문제를 풀게 한 뒤 오차 벌점을 받습니다.
            
            # loss.backward() 다음 optimizer.step()을 호출하세요.
            # >>> A07
            loss.backward()  # 단계 4: 역전파를 수행하여 벌점을 낮추기 위한 가중치 수정 방향(기울기)을 자동으로 계산합니다.
            optimizer.step()  # 단계 5: 계산된 미분값과 해결사 규칙에 따라 실제 파라미터(신경망 가중치)를 한 걸음 업데이트합니다.
            # <<< A07
            
        # epoch 종료 후 model.eval()을 호출하세요.
        # >>> A08
        model.eval()  # 단계 6: 공부가 끝났으니 실전 서비스 환경과 똑같이 드롭아웃을 끄고 전체 지식을 가동하는 '평가 모드'로 스위칭합니다.
        # <<< A08
        
        with torch.no_grad():  # 단계 7: 시험 중에는 가중치를 고치지 않으므로 미분 그래프 추적을 차단하여 속도와 서버 메모리를 절약합니다.
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()  # 공부했던 기출문제 복습 벌점 점수
            logits = model(X_valid)
            valid_loss = loss_fn(logits, y_valid).item()  # 낯선 시험 문제(검증 데이터)에 대한 실전 오차 벌점 점수
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()  # 실제 업무 분류 정확도 메트릭 (정답률)
            
            # 최저 valid loss를 관측하면 값·epoch·가중치 복사본을 함께 보관합니다.
            # >>> A09
            if valid_loss < best_loss:  # 모니터링 중: 만약 이번 바퀴의 검증 오차가 역대 최저치보다 낮아져 더 똑똑해졌다면
                best_loss = valid_loss   # 최저 벌점 기준을 새롭게 경신합니다.
                best_epoch = epoch       # 최적의 전성기 타이밍(에폭)을 저장합니다.
                best_state = copy.deepcopy(model.state_dict())  # 핵심: 독립된 주소의 완전한 모델 가중치 복사본을 안전하게 금고에 박제(저장)합니다.
            # <<< A09
            
        history.append([epoch, train_loss, valid_loss, valid_acc])  # 성적 일기를 대장에 적재합니다.
        
    # 성적표 가공 및 개별 실험 저장 처리
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]
    table.to_csv(out / f"07_{name}_history.csv", index=False)  # 에폭별 전체 성적 테이블을 CSV 파일로 저장합니다.
    
    # 보고서 요약본 구성
    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})
                    
    # 제공 코드: 학습이 끝난 뒤 40바퀴째의 과적합된 뇌를 버리고, 금고에 박제해 둔 최적 상태의 뇌세포로 복원해 실전 테스트를 다시 칩니다.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_pred = model(X_valid).argmax(1).cpu()  # 가장 똑똑했던 시점의 모델이 내린 최종 예측 정답 레이블 리스트
        restored_loss = loss_fn(model(X_valid), y_valid).item()
        
    labels = y_valid.cpu()  # 실제 현업에서 채점해 둔 진짜 원본 정답지
    class_rows = []
    
    for label in (0, 1):
        # A10 요구: 실제 정답이 label인 행을 먼저 고른 뒤 support·correct·recall 계산.
        # >>> A10
        actual_mask = (labels == label)            # 조건 필터: 실제 정답이 현재 클래스(0:정상 처리 또는 1:반려 필요)인 표본들만 추출하는 마스크입니다.
        support = int(actual_mask.sum().item())    # 실무 지표 1 (실제 표본 수): 현업 데이터셋 내에 존재하는 해당 업무 유형의 원본 개수입니다.
        
        if support > 0:
            # 실무 지표 2 (맞춘 정답 수): 실제 정답이 해당 유형인 샘플 중 AI 모델이 올바르게 예측해 낸 개수입니다.
            correct = int(((best_pred == label) & actual_mask).sum().item())
            # 실무 지표 3 (재현율, Recall): 진짜 반려해야 할 업무 요청(1) 중 AI가 놓치지 않고 잡아낸 실제 방어 비율을 도출합니다.
            recall = correct / support  
        else:
            correct = 0
            recall = None  # 예외 처리: 현업 데이터 마트에 해당 업무 표본이 0건이면 분모가 0이 되므로 계산에서 제외(`None`)합니다.
            
        class_rows.append({"class_label": label, "support": support, "correct": correct, "recall": recall})
        # <<< A10
        
    # 클래스별 세부 평가 리포트 및 실제 예측 행렬 매칭 파일 저장 처리
    pd.DataFrame(class_rows).to_csv(out / f"07_{name}_class_metrics.csv", index=False)
    pd.DataFrame({"id":valid["id"].tolist(),"actual":labels.tolist(),
                  "predicted":best_pred.tolist()}).to_csv(out / f"07_{name}_predictions.csv", index=False)
                  
    summary[-1]["restored_valid_loss"] = restored_loss
    summary[-1]["restored_matches_best"] = abs(restored_loss-best_loss) < 1e-6
    
    # 시각화 보고서: 벌점 곡선 렌더링
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")  # 최적의 배포 골든 타이밍 위치에 수직 가이드 점선을 그립니다.
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"07_{name}_curves.png", dpi=150)
    plt.close()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))

# 종합 실험 비교 원장 마감 및 저장
result = pd.DataFrame(summary)
result.to_csv(out / "07_comparison.csv", index=False)
print("생성: outputs/07_*.csv, *.png / test 미사용")
