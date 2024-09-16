import tensorflow as tf
import pandas as pd

if __name__ == "__main__":
    
    model_save_path = f'0db_KFold/model_ann_fold_4.h5'
    trained_model = tf.keras.models.load_model(model_save_path)
    test_dataset = pd.read_csv(f'0dB_KFold/test_fold_4.csv')

    # Make predictions
    label_dict = {
        0: 'normal (without faults)',
        1: 'pressure reduction in the intake manifold',
        2: 'compression ratio reduction',
        3: 'reduction of amount of fuel injected'
    }
    
    x = tf.convert_to_tensor(test_dataset.drop('class', axis=1))
    
    # x = tf.expand_dims(x, 0)
    predicted_classes = trained_model.predict(x).argmax(axis=1)
    
    # Map predictions to class labels
    predicted_labels = [label_dict[pred] for pred in predicted_classes]

    # # Output predictions
    for i, label in enumerate(predicted_labels):
        print(f"Sample {i}: {label}")
