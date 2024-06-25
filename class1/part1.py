import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer, TrainingArguments, AutoModelForSequenceClassification)

# Шаг 1: Загрузка данных из CSV файла
df = pd.read_csv('set.csv')

# Шаг 2: Преобразование данных в формат, совместимый с библиотекой datasets
dataset = Dataset.from_pandas(df[['text', 'textMistaked']])
dataset = dataset.rename_column('text', 'target_text')
dataset = dataset.rename_column('textMistaked', 'input_text')

# Шаг 3: Загрузка токенизатора и модели LLaMA для исправления ошибок
#tokenizer_corrector = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token = "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")
#model_corrector = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", token="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")
tokenizer_corrector = AutoTokenizer.from_pretrained("gpt2", token="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")
tokenizer_corrector.pad_token = tokenizer_corrector.eos_token  # Устанавливаем pad_token
model_corrector = AutoModelForCausalLM.from_pretrained("gpt2", token="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")
# Шаг 4: Функция токенизации данных для исправления ошибок
def tokenize_data_corrector(examples):
    inputs = examples['input_text']
    targets = examples['target_text']
    model_inputs = tokenizer_corrector(inputs, max_length=128, truncation=True, padding='max_length')
    labels = tokenizer_corrector(targets, max_length=128, truncation=True, padding='max_length').input_ids
    model_inputs['labels'] = labels
    return model_inputs

# Шаг 5: Применение токенизации к датасету для исправления ошибок
dataset = dataset.map(tokenize_data_corrector, batched=True)

# Шаг 6: Разделение на тренировочную и тестовую выборки для исправления ошибок
train_test_split_corrector = dataset.train_test_split(test_size=0.2)
train_dataset_corrector = train_test_split_corrector['train']
test_dataset_corrector = train_test_split_corrector['test']

# Шаг 7: Настройка параметров обучения для исправления ошибок
training_args_corrector = Seq2SeqTrainingArguments(
    output_dir='./results_corrector',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True
)

# Шаг 8: Создание и запуск тренера для исправления ошибок
trainer_corrector = Seq2SeqTrainer(
    model=model_corrector,
    args=training_args_corrector,
    train_dataset=train_dataset_corrector,
    eval_dataset=test_dataset_corrector,
    tokenizer=tokenizer_corrector
)

# Запуск обучения для исправления ошибок
trainer_corrector.train()

# Шаг 9: Оценка модели для исправления ошибок
results_corrector = trainer_corrector.evaluate()
print(f"Evaluation results for corrector: {results_corrector}")

# Шаг 10: Использование модели для исправления текста
def correct_text(text):
    inputs = tokenizer_corrector(text, return_tensors="pt", max_length=128, truncation=True, padding='max_length')
    outputs = model_corrector.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150 )
    corrected_text = tokenizer_corrector.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Исправление текста с ошибками
text_with_errors = "Ошыбки бывают разные: и пунктуационые, и речевые, но никогда не правильные"
corrected_text = correct_text(text_with_errors)
print("Исправленный текст:", corrected_text)
