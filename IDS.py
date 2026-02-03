!pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, MultiHeadAttention, LayerNormalization,
                                     Dense, Dropout, Conv1D, GlobalAveragePooling1D,
                                     Add)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

print("Loading datasets...")


normal_data = pd.read_csv('Normal_data.csv')
ovs_data = pd.read_csv('OVS.csv')
metasploitable_data = pd.read_csv('metasploitable-2.csv')


data = pd.concat([normal_data, ovs_data, metasploitable_data], ignore_index=True)



data['label'] = data['Label'].str.strip()


label_mapping = {
    'ddos': 'DDoS',
    'DDOS': 'DDoS',
    'dos': 'DoS',
    'DOS': 'DoS',
    'normal': 'Normal',
    'NORMAL': 'Normal',
    'probe': 'Probe',
    'PROBE': 'Probe',
    'bfa': 'BFA',
    'web-attack': 'Web-Attack',
    'WEB-ATTACK': 'Web-Attack',
    'botnet': 'BOTNET',
    'Botnet': 'BOTNET',
    'u2r': 'U2R'
}


data['label'] = data['label'].replace(label_mapping)


data = data.drop('Label', axis=1)

print(f"\nDataset shape: {data.shape}")
print(f"\nClass distribution (CLEANED LABELS!):")
class_dist = data['label'].value_counts().sort_values(ascending=False)
print(class_dist)
print(f"\nTotal unique classes: {data['label'].nunique()}")
print("\nExpected: 8 classes (DDoS, Probe, Normal, DoS, BFA, Web-Attack, BOTNET, U2R)")

print("\n" + "="*50)
print("Data Cleaning...")


data = data.loc[:, ~data.columns.str.contains('^Unnamed')]


print(f"Missing values before cleaning: {data.isnull().sum().sum()}")
data = data.dropna()


data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

print(f"Dataset shape after cleaning: {data.shape}")


X = data.drop('label', axis=1)
y = data['label']


X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

print(f"\nNumber of features: {X.shape[1]}")
print(f"Feature names: {list(X.columns)}")

print("\n" + "="*50)
print("Feature Selection (Top 6 features)...")


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


paper_features = ['Bwd Header Len', 'Flow Duration', 'Fwd Header Len',
                  'Flow Byts/s', 'Pkt Len Std', 'Pkt Size Avg']


if all(feat in X.columns for feat in paper_features):
    print("Using paper's exact 6 features")
    X_6 = X[paper_features].values
    selected_features = paper_features
else:
    
    print("Paper features not all available, using automatic selection")
    selector = SelectKBest(f_classif, k=6)
    X_6 = selector.fit_transform(X, y_encoded)
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices].tolist()

print(f"Selected features: {selected_features}")


if 'selector' in locals():
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)

    print("\nTop 10 feature scores:")
    print(feature_scores.head(10))
else:
    print("\nUsing paper's pre-selected features (no scoring needed)")

print("\n" + "="*50)
print("Standardizing features...")

scaler = StandardScaler()
X_6_scaled = scaler.fit_transform(X_6)


y_categorical = to_categorical(y_encoded)
num_classes = y_categorical.shape[1]

print(f"\nNumber of classes: {num_classes}")
print(f"Class names: {label_encoder.classes_}")

print("\n" + "="*50)
print("Splitting data...")

X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(
    X_6_scaled, y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"Training samples: {X_train_6.shape[0]}")
print(f"Testing samples: {X_test_6.shape[0]}")

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_encoding = self.add_weight(
            name='pos_encoding',
            shape=(1, input_shape[1], self.d_model),
            initializer='zeros',
            trainable=True
        )
        super(PositionalEncoding, self).build(input_shape)

    def call(self, x):
        return x + self.pos_encoding

def transformer_encoder_block(x, num_heads, d_model, dff, dropout_rate=0.1):
    """Transformer encoder block with proper residual connections"""
   
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([x, attn_output]))

    
    ffn_output = Dense(dff, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ffn_output]))

    return out2

def build_transformer_model(input_shape, num_classes,
                           d_model=64, num_heads=2,
                           num_transformer_blocks=2,
                           dff=128, dropout_rate=0.1):
    """Build transformer model for SDN intrusion detection"""
    inputs = Input(shape=input_shape)

    
    if len(input_shape) == 1:
        x = tf.expand_dims(inputs, axis=-1)
    else:
        x = inputs

    
    x = Dense(d_model)(x)

    
    x = PositionalEncoding(d_model)(x)
    x = Dropout(dropout_rate)(x)

   
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_block(x, num_heads, d_model, dff, dropout_rate)

    
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = LayerNormalization(epsilon=1e-6)(x)

    
    x = GlobalAveragePooling1D()(x)

    
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

print("\n" + "="*50)
print("Preparing data for transformer model...")


X_train = X_train_6.reshape((X_train_6.shape[0], X_train_6.shape[1], 1))
X_test = X_test_6.reshape((X_test_6.shape[0], X_test_6.shape[1], 1))

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train_6.shape}")
print(f"y_test shape: {y_test_6.shape}")

print("\n" + "="*50)
print("Building transformer model...")

model = build_transformer_model(
    input_shape=(X_train.shape[1], 1),
    num_classes=num_classes,
    d_model=64,
    num_heads=2,
    num_transformer_blocks=2,
    dff=128,
    dropout_rate=0.2
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall()]
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_transformer_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]


print("\n" + "="*50)
print("Training model...")

history = model.fit(
    X_train, y_train_6,
    epochs=50,
    batch_size=128,
    validation_data=(X_test, y_test_6),
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*50)
print("Evaluating model...")

results = model.evaluate(X_test, y_test_6, verbose=0)
print(f"\nTest Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]*100:.2f}%")
print(f"Test Precision: {results[2]*100:.2f}%")
print(f"Test Recall: {results[3]*100:.2f}%")


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_6, axis=1)


cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.title("Confusion Matrix - Transformer Model (6 features)", fontsize=14)
plt.tight_layout()
plt.show()


print("\n" + "="*50)
print("Classification Report:")
print(classification_report(y_true, y_pred_classes,
                          target_names=label_encoder.classes_,
                          digits=4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))


axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)


axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Model Loss', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("Saving model and preprocessing artifacts...")


model.save('sdn_transformer_model_final.h5')


import pickle

artifacts = {
    'scaler': scaler,
    'label_encoder': label_encoder,
    'selected_features': selected_features,
}

if 'selector' in locals():
    artifacts['selector'] = selector

with open('preprocessing_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("Model and artifacts saved successfully!")
print("\nFiles created:")
print("- best_transformer_model.h5 (best model during training)")
print("- sdn_transformer_model_final.h5 (final model)")
print("- preprocessing_artifacts.pkl (scaler, encoder, feature names)")


import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf
import json
import logging

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional Encoding layer - matches training code"""
    def __init__(self, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_encoding = self.add_weight(
            name='pos_encoding',
            shape=(1, input_shape[1], self.d_model),
            initializer='zeros',
            trainable=True
        )
        super(PositionalEncoding, self).build(input_shape)

    def call(self, x):
        return x + self.pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config

class IDPSConfig:
    """Configuration for IDPS system"""
   
    MODEL_PATH = 'best_transformer_model.h5'
    PREPROCESSING_PATH = 'preprocessing_artifacts.pkl'

   =
    DOS_BLOCK_DURATION = 3600  
    DDOS_BLOCK_DURATION = 7200  
    PROBE_BLOCK_DURATION = -1  
    BFA_BLOCK_DURATION = 300   
    WEB_ATTACK_BLOCK_DURATION = 1800  
    BOTNET_BLOCK_DURATION = -1  
    U2R_BLOCK_DURATION = -1  

    
    DDOS_RATE_LIMIT = 1000  
    NORMAL_RATE_LIMIT = 10000

    
    ALERT_LEVELS = {
        'Normal': 0,
        'DoS': 2,
        'DDoS': 3,
        'Probe': 1,
        'BFA': 2,
        'Web-Attack': 2,
        'BOTNET': 3,
        'U2R': 4  
    }

    
    HONEYPOT_IP = '192.168.100.99'
    HONEYPOT_PORT = 8888

   
    QUARANTINE_VLAN = 999

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('idps.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('IDPS')

class AttackDetector:
    """Detects attacks using trained Transformer model"""

    def __init__(self):
        logger.info("Loading detection model...")

       
        custom_objects = {'PositionalEncoding': PositionalEncoding}
        self.model = load_model(IDPSConfig.MODEL_PATH, custom_objects=custom_objects)

        
        with open(IDPSConfig.PREPROCESSING_PATH, 'rb') as f:
            artifacts = pickle.load(f)
            self.scaler = artifacts['scaler']
            self.label_encoder = artifacts['label_encoder']
            self.selected_features = artifacts['selected_features']

        logger.info(f"Model loaded. Classes: {self.label_encoder.classes_}")
        logger.info(f"Using features: {self.selected_features}")

    def extract_features(self, traffic_data):
        """Extract the 6 features used by the model"""
       
        if isinstance(traffic_data, dict):
            features = [traffic_data[feat] for feat in self.selected_features]
        else:
            features = traffic_data[self.selected_features].values

        return np.array(features).reshape(1, -1)

    def predict(self, traffic_data):
        """Predict attack type from traffic data"""
        
        features = self.extract_features(traffic_data)

        
        features_scaled = self.scaler.transform(features)

        
        features_reshaped = features_scaled.reshape(1, 6, 1)

       
        prediction = self.model.predict(features_reshaped, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

       
        attack_type = self.label_encoder.classes_[predicted_class]

        return {
            'attack_type': attack_type,
            'confidence': float(confidence),
            'probabilities': {self.label_encoder.classes_[i]: float(prediction[0][i])
                            for i in range(len(self.label_encoder.classes_))}
        }

class MitigationEngine:
    """Applies countermeasures based on attack type"""

    def __init__(self):
        self.active_blocks = {}  
        self.attack_stats = {attack: 0 for attack in IDPSConfig.ALERT_LEVELS.keys()}
        logger.info("Mitigation Engine initialized")

    def apply_countermeasure(self, attack_type, traffic_info, detection_result):
        """Apply appropriate countermeasure based on attack type"""

        src_ip = traffic_info.get('src_ip', 'unknown')
        dst_ip = traffic_info.get('dst_ip', 'unknown')

        
        self.attack_stats[attack_type] += 1

       
        alert_level = IDPSConfig.ALERT_LEVELS.get(attack_type, 0)

        logger.info(f" Attack Detected: {attack_type} | Source: {src_ip} | "
                   f"Confidence: {detection_result['confidence']:.2%} | "
                   f"Alert Level: {alert_level}")

        
        if attack_type == "Normal":
            return self._allow_traffic(traffic_info)

        elif attack_type == "DoS":
            return self._block_dos(src_ip, traffic_info)

        elif attack_type == "DDoS":
            return self._block_ddos(src_ip, traffic_info)

        elif attack_type == "Probe":
            return self._redirect_to_honeypot(src_ip, traffic_info)

        elif attack_type == "BFA":
            return self._temporary_block(src_ip, traffic_info, IDPSConfig.BFA_BLOCK_DURATION)

        elif attack_type == "Web-Attack":
            return self._enable_waf(src_ip, traffic_info)

        elif attack_type == "BOTNET":
            return self._quarantine_host(src_ip, traffic_info)

        elif attack_type == "U2R":
            return self._critical_response(src_ip, traffic_info)

        else:
            return self._default_block(src_ip, traffic_info)

    def _allow_traffic(self, traffic_info):
        """Allow normal traffic"""
        return {
            'action': 'ALLOW',
            'flow_rule': {
                'priority': 100,
                'match': {},
                'actions': ['OUTPUT:NORMAL']
            },
            'description': 'Normal traffic - allowed'
        }

    def _block_dos(self, src_ip, traffic_info):
        """Block DoS attack"""
        self.active_blocks[src_ip] = {
            'attack_type': 'DoS',
            'blocked_at': datetime.now(),
            'duration': IDPSConfig.DOS_BLOCK_DURATION
        }

        return {
            'action': 'BLOCK',
            'flow_rule': {
                'priority': 1000,
                'match': {'ipv4_src': src_ip},
                'actions': ['DROP'],
                'idle_timeout': IDPSConfig.DOS_BLOCK_DURATION
            },
            'description': f'DoS attack blocked from {src_ip} for {IDPSConfig.DOS_BLOCK_DURATION}s'
        }

    def _block_ddos(self, src_ip, traffic_info):
        """Block DDoS attack with rate limiting"""
        self.active_blocks[src_ip] = {
            'attack_type': 'DDoS',
            'blocked_at': datetime.now(),
            'duration': IDPSConfig.DDOS_BLOCK_DURATION
        }

        return {
            'action': 'BLOCK_AND_RATE_LIMIT',
            'flow_rule': {
                'priority': 1000,
                'match': {'ipv4_src': src_ip},
                'actions': ['DROP'],
                'idle_timeout': IDPSConfig.DDOS_BLOCK_DURATION
            },
            'rate_limit': {
                'max_rate': IDPSConfig.DDOS_RATE_LIMIT,
                'burst_size': 100
            },
            'description': f'DDoS attack blocked from {src_ip} with rate limiting'
        }

    def _redirect_to_honeypot(self, src_ip, traffic_info):
        """Redirect probe/scan to honeypot"""
        self.active_blocks[src_ip] = {
            'attack_type': 'Probe',
            'blocked_at': datetime.now(),
            'duration': -1  
        }

        return {
            'action': 'REDIRECT_HONEYPOT',
            'flow_rule': {
                'priority': 900,
                'match': {'ipv4_src': src_ip},
                'actions': [
                    f'SET_FIELD:ipv4_dst={IDPSConfig.HONEYPOT_IP}',
                    f'SET_FIELD:tcp_dst={IDPSConfig.HONEYPOT_PORT}',
                    'OUTPUT:CONTROLLER'
                ]
            },
            'description': f'Probe attack from {src_ip} redirected to honeypot'
        }

    def _temporary_block(self, src_ip, traffic_info, duration):
        """Temporary block (for BFA)"""
        self.active_blocks[src_ip] = {
            'attack_type': 'BFA',
            'blocked_at': datetime.now(),
            'duration': duration
        }

        return {
            'action': 'TEMP_BLOCK',
            'flow_rule': {
                'priority': 800,
                'match': {'ipv4_src': src_ip},
                'actions': ['DROP'],
                'idle_timeout': duration
            },
            'description': f'BFA from {src_ip} temporarily blocked for {duration}s'
        }

    def _enable_waf(self, src_ip, traffic_info):
        """Enable Web Application Firewall rules"""
        self.active_blocks[src_ip] = {
            'attack_type': 'Web-Attack',
            'blocked_at': datetime.now(),
            'duration': IDPSConfig.WEB_ATTACK_BLOCK_DURATION
        }

        return {
            'action': 'WAF_INSPECT',
            'flow_rule': {
                'priority': 950,
                'match': {
                    'ipv4_src': src_ip,
                    'eth_type': 0x0800,
                    'ip_proto': 6  
                },
                'actions': ['OUTPUT:CONTROLLER']  
            },
            'waf_rules': [
                'block_sql_injection',
                'block_xss',
                'block_path_traversal'
            ],
            'description': f'Web attack from {src_ip} - WAF enabled'
        }

    def _quarantine_host(self, src_ip, traffic_info):
        """Quarantine botnet-infected host"""
        self.active_blocks[src_ip] = {
            'attack_type': 'BOTNET',
            'blocked_at': datetime.now(),
            'duration': -1  
        }

        return {
            'action': 'QUARANTINE',
            'flow_rule': {
                'priority': 1000,
                'match': {'ipv4_src': src_ip},
                'actions': [
                    f'SET_FIELD:vlan_vid={IDPSConfig.QUARANTINE_VLAN}',
                    'OUTPUT:QUARANTINE_PORT'
                ]
            },
            'description': f'Botnet-infected host {src_ip} quarantined to VLAN {IDPSConfig.QUARANTINE_VLAN}'
        }

    def _critical_response(self, src_ip, traffic_info):
        """Critical response for U2R attacks"""
        self.active_blocks[src_ip] = {
            'attack_type': 'U2R',
            'blocked_at': datetime.now(),
            'duration': -1  
        }

       
        self._send_critical_alert(src_ip, traffic_info)

        return {
            'action': 'CRITICAL_BLOCK',
            'flow_rule': {
                'priority': 2000, 
                'match': {'ipv4_src': src_ip},
                'actions': ['DROP']
            },
            'additional_actions': [
                'KILL_EXISTING_CONNECTIONS',
                'ALERT_ADMIN',
                'FORENSIC_CAPTURE'
            ],
            'description': f' CRITICAL: U2R attack from {src_ip} - immediate block and alert'
        }

    def _default_block(self, src_ip, traffic_info):
        """Default blocking action"""
        return {
            'action': 'BLOCK',
            'flow_rule': {
                'priority': 500,
                'match': {'ipv4_src': src_ip},
                'actions': ['DROP']
            },
            'description': f'Unknown attack type - default block applied to {src_ip}'
        }

    def _send_critical_alert(self, src_ip, traffic_info):
        """Send critical alert to administrators"""
        alert = {
            'level': 'CRITICAL',
            'attack_type': 'U2R',
            'source_ip': src_ip,
            'timestamp': datetime.now().isoformat(),
            'message': f'Privilege escalation attempt detected from {src_ip}'
        }
        logger.critical(f" CRITICAL ALERT: {alert}")
       

    def get_statistics(self):
        """Get attack statistics"""
        return {
            'total_attacks': sum(self.attack_stats.values()),
            'attack_breakdown': self.attack_stats,
            'active_blocks': len(self.active_blocks),
            'blocked_ips': list(self.active_blocks.keys())
        }

class SDNController:
    """Interface to SDN controller (OpenFlow)"""

    def __init__(self, controller_ip='127.0.0.1', controller_port=6653):
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.installed_rules = []
        logger.info(f"SDN Controller interface initialized: {controller_ip}:{controller_port}")

    def install_flow_rule(self, flow_rule):
        """Install OpenFlow rule on SDN switches"""

       

        rule_id = len(self.installed_rules) + 1
        flow_rule['rule_id'] = rule_id
        flow_rule['installed_at'] = datetime.now().isoformat()

        self.installed_rules.append(flow_rule)

        logger.info(f" Flow rule #{rule_id} installed: {flow_rule.get('description', 'No description')}")
        logger.debug(f"Rule details: {json.dumps(flow_rule, indent=2)}")

        return rule_id

    def remove_flow_rule(self, rule_id):
        """Remove flow rule"""
        self.installed_rules = [r for r in self.installed_rules if r['rule_id'] != rule_id]
        logger.info(f" Flow rule #{rule_id} removed")

    def get_active_rules(self):
        """Get all active flow rules"""
        return self.installed_rules

    def clear_all_rules(self):
        """Clear all flow rules"""
        count = len(self.installed_rules)
        self.installed_rules = []
        logger.warning(f"⚠️ All flow rules cleared ({count} rules removed)")

class IDPS:
    """Complete Intrusion Detection and Prevention System"""

    def __init__(self):
        logger.info("="*60)
        logger.info("Initializing SDN IDPS System")
        logger.info("="*60)

        self.detector = AttackDetector()
        self.mitigator = MitigationEngine()
        self.sdn_controller = SDNController()

        self.total_processed = 0
        self.start_time = datetime.now()

        logger.info(" IDPS System ready!")
        logger.info("="*60)

    def process_traffic(self, traffic_data):
        """Process incoming traffic through complete IDPS pipeline"""

        self.total_processed += 1

        
        detection_result = self.detector.predict(traffic_data)
        attack_type = detection_result['attack_type']
        confidence = detection_result['confidence']

        
        if attack_type != "Normal" or confidence < 0.95:
            mitigation_action = self.mitigator.apply_countermeasure(
                attack_type,
                traffic_data,
                detection_result
            )

          
            if mitigation_action['action'] != 'ALLOW':
                rule_id = self.sdn_controller.install_flow_rule(
                    mitigation_action['flow_rule']
                )
                mitigation_action['rule_id'] = rule_id

            return {
                'timestamp': datetime.now().isoformat(),
                'traffic_id': self.total_processed,
                'detection': detection_result,
                'mitigation': mitigation_action,
                'status': 'MITIGATED'
            }

        
        return {
            'timestamp': datetime.now().isoformat(),
            'traffic_id': self.total_processed,
            'detection': detection_result,
            'status': 'ALLOWED'
        }

    def get_dashboard_stats(self):
        """Get dashboard statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            'system_info': {
                'uptime_seconds': uptime,
                'total_traffic_processed': self.total_processed,
                'traffic_rate': self.total_processed / max(uptime, 1)
            },
            'attack_stats': self.mitigator.get_statistics(),
            'sdn_info': {
                'active_flow_rules': len(self.sdn_controller.get_active_rules()),
                'controller_status': 'CONNECTED'
            }
        }

    def print_dashboard(self):
        """Print dashboard to console"""
        stats = self.get_dashboard_stats()

        print("\n" + "="*60)
        print(" IDPS DASHBOARD")
        print("="*60)
        print(f"  Uptime: {stats['system_info']['uptime_seconds']:.0f}s")
        print(f" Total Traffic: {stats['system_info']['total_traffic_processed']}")
        print(f" Rate: {stats['system_info']['traffic_rate']:.2f} packets/sec")
        print("\n Attack Statistics:")
        for attack, count in stats['attack_stats']['attack_breakdown'].items():
            if count > 0:
                print(f"   {attack:15s}: {count:6d}")
        print(f"\n Active Blocks: {stats['attack_stats']['active_blocks']}")
        print(f" Flow Rules: {stats['sdn_info']['active_flow_rules']}")
        print("="*60 + "\n")

def demo_idps():
    """Demonstrate IDPS system with sample traffic"""

    
    idps = IDPS()

    
    test_scenarios = [
        {
            'name': 'Normal Traffic',
            'data': {
                'Bwd Header Len': 40,
                'Flow Duration': 120000,
                'Fwd Header Len': 40,
                'Flow Byts/s': 8000,
                'Pkt Len Std': 100,
                'Pkt Size Avg': 500,
                'src_ip': '192.168.1.100',
                'dst_ip': '10.0.0.50'
            }
        },
        {
            'name': 'DoS Attack (High packet rate, short duration)',
            'data': {
                'Bwd Header Len': 0,
                'Flow Duration': 1000,
                'Fwd Header Len': 20,
                'Flow Byts/s': 500000,
                'Pkt Len Std': 0,
                'Pkt Size Avg': 40,
                'src_ip': '198.51.100.10',
                'dst_ip': '10.0.0.50'
            }
        },
        {
            'name': 'DDoS Attack (Massive traffic)',
            'data': {
                'Bwd Header Len': 0,
                'Flow Duration': 500,
                'Fwd Header Len': 20,
                'Flow Byts/s': 2000000,
                'Pkt Len Std': 5,
                'Pkt Size Avg': 64,
                'src_ip': '203.0.113.45',
                'dst_ip': '10.0.0.50'
            }
        },
        {
            'name': 'Port Scan (Probe)',
            'data': {
                'Bwd Header Len': 0,
                'Flow Duration': 100,
                'Fwd Header Len': 20,
                'Flow Byts/s': 200,
                'Pkt Len Std': 0,
                'Pkt Size Avg': 40,
                'src_ip': '198.51.100.23',
                'dst_ip': '10.0.0.50'
            }
        },
        {
            'name': 'Brute Force Attack (BFA)',
            'data': {
                'Bwd Header Len': 20,
                'Flow Duration': 5000,
                'Fwd Header Len': 32,
                'Flow Byts/s': 15000,
                'Pkt Len Std': 10,
                'Pkt Size Avg': 80,
                'src_ip': '203.0.113.88',
                'dst_ip': '10.0.0.50'
            }
        },
        {
            'name': 'Web Attack (SQL Injection attempt)',
            'data': {
                'Bwd Header Len': 60,
                'Flow Duration': 8000,
                'Fwd Header Len': 200,
                'Flow Byts/s': 50000,
                'Pkt Len Std': 150,
                'Pkt Size Avg': 800,
                'src_ip': '198.51.100.99',
                'dst_ip': '10.0.0.50'
            }
        }
    ]

    print("\n Starting IDPS Demo...\n")
    print("NOTE: Model predictions depend on training data distribution.")
    print("If all traffic is classified as 'Normal', try adjusting feature values")
    print("or use real network traffic data for testing.\n")

    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"Testing: {scenario['name']}")
        print(f"{'='*60}")

        result = idps.process_traffic(scenario['data'])

        print(f"\n Result:")
        print(f"  Attack Type: {result['detection']['attack_type']}")
        print(f"  Confidence: {result['detection']['confidence']*100:.2f}%")
        print(f"  Status: {result['status']}")

        
        probs = result['detection']['probabilities']
        top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n  Top 3 Predictions:")
        for attack, prob in top_3:
            print(f"    {attack:15s}: {prob*100:.4f}%")

        if result['status'] == 'MITIGATED':
            print(f"\n    Mitigation: {result['mitigation']['action']}")
            print(f"   {result['mitigation']['description']}")

        time.sleep(1)


    idps.print_dashboard()

    print("\n Demo completed!")
    print("\n Tips for better results:")
    print("  - Use real network traffic data for testing")
    print("  - Ensure feature values match training data scale")
    print("  - Check preprocessing artifacts for feature statistics")
    print("  - Model accuracy: Check training results for expected performance")

if __name__ == "__main__":
    demo_idps()