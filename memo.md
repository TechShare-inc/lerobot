Datasetの重要な責務
global index
    ↓
episodeとframeを特定
    ↓
Parquetからstate/actionを取得
    ↓
動画から画像を取得
    ↓
delta_timestampsに基づいて前後フレームを取得
    ↓
Tensor辞書として返す

4. processor
外部表現
dict[str, Tensor]
    ↓
Transition
    ↓
複数のProcessorStep
    ↓
Policy用dict


batch_processor.py : batch単位の処理

normalize_processor.py : state/actionの正規化と逆正規化

device_processor.py : CPU/GPUへの移動

observation_processor.py : observationの整形

rename_processor.py : feature keyの変換

pipeline.py : ProcessorStepを順番に実行

converters.py : dict、Transition、PolicyAction間の変換

factory.py : 

Preprocessor
  生のobservation/action
    → Tensor化
    → batch化
    → device移動
    → 正規化
    → Policy入力

Postprocessor
  Policy出力
    → action切り出し
    → 逆正規化
    → 実環境用action







TrainPipelineConfig
       │
       ├── DatasetConfig
       │       ↓
       │   make_dataset()
       │       ↓
       │   LeRobotDataset
       │
       └── PolicyConfig
               ↓
           make_policy()
               ↓
             Policy

Dataset metadata
       ├── feature shape → PolicyConfig
       └── stats → Processor

DataLoader
   ↓
batch
   ↓
Preprocessor
   ↓
Policy.forward()
   ↓
loss
   ↓
backward
   ↓
Optimizer












checkpoint directory
├── config.json
├── model.safetensors
├── policy_preprocessor.json
├── processor用safetensors
├── policy_postprocessor.json
└── processor用safetensors

observation
    ↓
Preprocessor
    ↓
Policy.select_action()
    ↓
action queue
    ↓
Postprocessor
    ↓
実値action










