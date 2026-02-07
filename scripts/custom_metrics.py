"""
Custom metrics for Writer Identification Model
"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class MacroPrecision(tf.keras.metrics.Metric):
    """Macro-averaged Precision metric"""
    
    def __init__(self, num_classes, name='macro_precision', **kwargs):
        super(MacroPrecision, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name='tp', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )
        self.false_positives = self.add_weight(
            name='fp', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions and true labels to class indices
        y_pred_indices = tf.argmax(y_pred, axis=-1)
        y_true_indices = tf.argmax(y_true, axis=-1)

        # One-hot encode the indices
        y_pred_one_hot = tf.one_hot(y_pred_indices, depth=self.num_classes)
        y_true_one_hot = tf.one_hot(y_true_indices, depth=self.num_classes)

        # Calculate true positives and false positives per class
        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fp = tf.reduce_sum(y_pred_one_hot * (1 - y_true_one_hot), axis=0)

        # Update the state variables
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)

    def result(self):
        # Compute per-class precision and macro-average it
        precision_per_class = self.true_positives / (
            self.true_positives + self.false_positives + tf.keras.backend.epsilon()
        )
        macro_precision = tf.reduce_mean(precision_per_class)
        return macro_precision

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))

    def get_config(self):
        base_config = super(MacroPrecision, self).get_config()
        base_config.update({'num_classes': self.num_classes})
        return base_config


@tf.keras.utils.register_keras_serializable()
class MacroRecall(tf.keras.metrics.Metric):
    """Macro-averaged Recall metric"""
    
    def __init__(self, num_classes, name='macro_recall', **kwargs):
        super(MacroRecall, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name='tp', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )
        self.false_negatives = self.add_weight(
            name='fn', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions and true labels to class indices
        y_pred_indices = tf.argmax(y_pred, axis=-1)
        y_true_indices = tf.argmax(y_true, axis=-1)

        # One-hot encode the indices
        y_pred_one_hot = tf.one_hot(y_pred_indices, depth=self.num_classes)
        y_true_one_hot = tf.one_hot(y_true_indices, depth=self.num_classes)

        # Calculate true positives and false negatives per class
        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fn = tf.reduce_sum(y_true_one_hot * (1 - y_pred_one_hot), axis=0)

        # Update the state variables
        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)

    def result(self):
        # Compute per-class recall and macro-average it
        recall_per_class = self.true_positives / (
            self.true_positives + self.false_negatives + tf.keras.backend.epsilon()
        )
        macro_recall = tf.reduce_mean(recall_per_class)
        return macro_recall

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

    def get_config(self):
        base_config = super(MacroRecall, self).get_config()
        base_config.update({'num_classes': self.num_classes})
        return base_config


@tf.keras.utils.register_keras_serializable()
class MacroF1Score(tf.keras.metrics.Metric):
    """Macro-averaged F1 Score metric"""
    
    def __init__(self, num_classes, name='macro_f1_score', **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precision_metric = MacroPrecision(num_classes)
        self.recall_metric = MacroRecall(num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(y_true, y_pred)
        self.recall_metric.update_state(y_true, y_pred)

    def result(self):
        # Compute the macro-averaged precision and recall
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()
        # Compute the macro-averaged F1 score
        f1_score = 2 * (precision * recall) / (
            precision + recall + tf.keras.backend.epsilon()
        )
        return f1_score

    def reset_states(self):
        self.precision_metric.reset_states()
        self.recall_metric.reset_states()

    def get_config(self):
        base_config = super(MacroF1Score, self).get_config()
        base_config.update({'num_classes': self.num_classes})
        return base_config
