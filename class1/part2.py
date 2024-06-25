import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer, TrainingArguments, AutoModelForSequenceClassification)

# Шаг 11: Загрузка данных для определения стиля текста
style_df = pd.read_csv('dte.csv')  # CSV файл с данными для определения стиля текста
style_dataset = Dataset.from_pandas(style_df)

# Шаг 12: Загрузка токенизатора и модели для определения стиля текста
tokenizer_style = AutoTokenizer.from_pretrained("gpt2", token = "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")
model_style = AutoModelForSequenceClassification.from_pretrained("gpt2", token="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")

# Шаг 13: Функция токенизации данных для определения стиля текста
def tokenize_data_style(examples):
    inputs = examples['text']
    model_inputs = tokenizer_style(inputs, truncation=True, padding='max_length')
    return model_inputs

# Шаг 14: Применение токенизации к датасету для определения стиля текста
style_dataset = style_dataset.map(tokenize_data_style, batched=True)

# Шаг 15: Разделение на тренировочную и тестовую выборки для определения стиля текста
train_test_split_style = style_dataset.train_test_split(test_size=0.2)
train_dataset_style = train_test_split_style['train']
test_dataset_style = train_test_split_style['test']

# Шаг 16: Настройка параметров обучения для определения стиля текста
training_args_style = TrainingArguments(
    output_dir='./results_style',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3
)

# Шаг 17: Создание и запуск тренера для определения стиля текста
trainer_style = Trainer(
    model=model_style,
    args=training_args_style,
    train_dataset=train_dataset_style,
    eval_dataset=test_dataset_style,
    tokenizer=tokenizer_style
)

# Запуск обучения для определения стиля текста
trainer_style.train()

# Шаг 18: Оценка модели для определения стиля текста
results_style = trainer_style.evaluate()
print(f"Evaluation results for style: {results_style}")

# Шаг 19: Использование модели для определения стиля текста
def predict_style(text):
    inputs = tokenizer_style(text, return_tensors="pt", truncation=True, padding='max_length')
    outputs = model_style(inputs['input_ids'], inputs['attention_mask'])
    predicted_style = outputs.logits.argmax(dim=1).item()
    return predicted_style
'''
# Шаг 20: Выбор файла пользователем
file_path = input("Введите путь к файлу для проверки: ")

# Шаг 21: Чтение текста из файла
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Шаг 22: Исправление ошибок в тексте
corrected_text = correct_text(text)

# Шаг 23: Определение стиля исправленного текста
predicted_style = predict_style(corrected_text)
print(f"Исправленный текст: {corrected_text}")
print(f"Определенный стиль текста: {predicted_style}")
'''
# Исправление текста с ошибками
text_with_errors = "Ошыбки бывают разные: и пунктуационые, и речевые, но никогда не правильные"
predicted_style = predict_style(text_with_errors)
print(f"Определенный стиль текста: {predicted_style}")