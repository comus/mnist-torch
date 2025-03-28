1. 我用的是 mac, 蘋果芯片
2. 所以我一開始是叫他用 mlx 做
3. 另外已經安裝了 miniconda
4. 之後我發現，其實用 pytorch 做會比較好，而且 pytorch 也支持蘋果芯片，他有什麼 MPS 的加速，所以我就改用 pytorch 做

我是這樣問 AI 開始的

https://poe.com/s/j2pKS1Mpmj6Fdt4a31PD

```
conda create -n mlx-distill python=3.10 -y
conda activate mlx-distill

pip install numpy scikit-learn matplotlib
pip install mlx
pip install pandas
pip install seaborn
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install ultralytics
pip install tqdm pyyaml
```

# 訓練教師模型

```
python train_teacher.py
```

大約會有這樣的輸出

```
=== 教師模型訓練結果 ===
訓練時間: 11.31 秒
最終準確率: 0.9902
參數數量: 225,034
模型已保存到 models/teacher_model.pkl
訓練進度圖表已保存到 results/training_progress.png
預測樣本圖表已保存到 results/prediction_samples.png
混淆矩陣已保存到 results/confusion_matrix.png
```

# 蒸餾學生模型

```
python distill_student.py
```

大約會有這樣的輸出

```
Epoch 5/5, Loss: 2.0956, Student Accuracy: 0.9843, Teacher Accuracy: 0.9902

=== 知識蒸餾結果 ===
蒸餾時間: 13.35 秒
學生模型準確率: 0.9843
教師模型準確率: 0.9902
學生模型已保存到 models/student_model.pkl

=== 模型對比 ===
教師模型參數數量: 225,034
學生模型參數數量: 173,930
縮小比例: 1.29x

=== 推理速度對比 ===
Input tensor shape in model: (100, 1, 28, 28)
Input tensor shape in model: (100, 1, 28, 28)
Input tensor shape in model: (100, 1, 28, 28)
Input tensor shape in model: (100, 1, 28, 28)
Input tensor shape in model: (100, 1, 28, 28)
Input tensor shape in model: (100, 1, 28, 28)
Input tensor shape in model: (100, 1, 28, 28)
Input tensor shape in model: (100, 1, 28, 28)
Input tensor shape in model: (100, 1, 28, 28)
Input tensor shape in model: (100, 1, 28, 28)
教師模型推理速度: 0.0001 秒
學生模型推理速度: 0.0000 秒
速度提升: 2.06x
/Users/region/mlx_distillation/utils.py:88: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
  plt.tight_layout()
正在生成學習曲線圖...
正在生成學生與教師的預測比較圖...
Input tensor shape in model: (10, 1, 28, 28)
正在生成學生模型的混淆矩陣...
正在分析困難案例...
Input tensor shape in model: (14000, 1, 28, 28)
Input tensor shape in model: (10, 1, 28, 28)

=== 視覺化結果 ===
模型比較圖表已保存到 results/distillation_results.png
學習曲線已保存到 results/student_learning_curves.png
學生與教師的預測比較已保存到 results/student_teacher_comparison.png
學生模型的混淆矩陣已保存到 results/student_confusion_matrix.png
困難案例分析已保存到 results/hard_cases.png
```
