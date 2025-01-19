import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import random

def parametre_hesaplama(y_test,y_pred, z):

    # Başlangıç değerleri
    TP = 0  # Gerçek pozitif
    TN = 0  # Gerçek negatif
    FP = 0  # Yanlış pozitif
    FN = 0  # Yanlış negatif
    
    # Her bir tahmini değerlendir
    for gercek, tahmin in zip(y_test, y_pred):
        if gercek == 1 and tahmin == 1:
            TP += 1  # Gerçek pozitif
        elif gercek == 0 and tahmin == 0:
            TN += 1  # Gerçek negatif
        elif gercek == 0 and tahmin == 1:
            FP += 1  # Yanlış pozitif
        elif gercek == 1 and tahmin == 0:
            FN += 1  # Yanlış negatif

    N = TP + TN + FP + FN

    #--------dogruluk------
    dogruluk = (TP + TN) / N

    #--------duyarlilik----
    duyarlilik = TP / (TP + FN)

    #--------ozgulluk------
    ozgulluk = TN / (TN + FP)
    
    #--------f_skor--------
    f_skor = 2 * (ozgulluk * duyarlilik) / (ozgulluk + duyarlilik)

    #--------kappa---------
    Ppositive = (TP + FN) * (TP + FP) / (N ** 2)
    Pnegative = (TN + FP) * (TN + FN) / (N ** 2)
    P_e = Ppositive + Pnegative
    P_o = dogruluk

    kappa = (P_o - P_e) / (1 - P_e)

    #------------performans parametreleri------------
    # Metrikleri ve değerlerini hazırlama
    metrics = {
                "Doğruluk (Accuracy)": dogruluk,
                "Duyarlılık (Sensitivity)": duyarlilik,
                "Özgüllük (Specificity)": ozgulluk,
                "F-Skor (F-Score)": f_skor,
                "Kappa (Cohen's Kappa)": kappa,
              }

    # Matplotlib penceresi
    plt.figure(figsize=(6, 4))
    plt.title("Model Performans Metrikleri\n("+z+")", fontsize=14, weight="bold")
    plt.axis("off")  # Grafik eksenlerini kapat
    # Her bir metriği liste olarak yazdırma
    for i, (name, value) in enumerate(metrics.items(), start=1):
        plt.text(0.1, 1 - i * 0.15, f"{name}: {value:.2f}", fontsize=12)
    plt.show()

    #--------karmasiklik_matrisi---------
    # Karışıklık matrisindeki FP, FN, TP, TN değerlerini ekleme
    labels = [
        ['TN\n{}'.format(TN), 'FP\n{}'.format(FP)],
        ['FN\n{}'.format(FN), 'TP\n{}'.format(TP)]
    ]

    cm = ((TN,FP),(FN,TP))
    # Heatmap çizdirme
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                xticklabels=['Düşük Kalite', 'Yüksek Kalite'], 
                yticklabels=['Düşük Kalite', 'Yüksek Kalite'])

    plt.title('Karmaşıklık Matrisi ('+z+')')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.show()

    #---------------------ROC Eğrisi için TPR ve FPR Hesaplama---------------------
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    #AUC Değerini Hesaplama
    roc_auc = roc_auc_score(y_test, y_pred)

    #ROC Eğrisini Çizme
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Eğrisi (AUC = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='Şans Çizgisi')
    plt.title('ROC Eğrisi ('+z+')')
    plt.xlabel('Yanlış Pozitif Oranı (FPR)')
    plt.ylabel('Doğru Pozitif Oranı (TPR)')
    plt.legend()
    plt.grid()
    plt.show()


#-------------------------------------------------------KNN-------------------------------------------------------
def knn_predict(X_train, y_train, X_test, k):
    predictions = []  # Tahmin sonuçları
    
    for test_point in X_test:
        # Tüm eğitim noktalarıyla uzaklıkları hesapla
        distances = []
        for i, train_point in enumerate(X_train):
            distance = euclidean_distance(test_point, train_point)
            distances.append((distance, y_train[i]))
        
        # Uzaklığa göre sırala
        distances.sort(key=lambda x: x[0])
        
        # En yakın k komşunun sınıflarını al
        k_neighbors = [label for _, label in distances[:k]]
        
        # En sık görülen sınıfı tahmin olarak belirle
        most_common = Counter(k_neighbors).most_common(1)[0][0]
        predictions.append(most_common)
    
    return predictions

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


#-------------------------------------------------------KARAR AĞACI-------------------------------------------------------
# Gini İndeksi Hesaplama Fonksiyonu
def gini_index(y):
    class_counts = Counter(y)
    total = len(y)
    gini = 1 - sum((count / total) ** 2 for count in class_counts.values())
    return gini

# En iyi bölmeyi bulma
def best_split(X, y):
    best_gini = float('inf')
    best_feature = None
    best_threshold = None
    for feature in range(X.shape[1]):  # Her özellik için
        thresholds = np.unique(X[:, feature])  # Özelliğin farklı değerlerini al
        for threshold in thresholds:  # Her eşik değeri için
            left_mask = X[:, feature] <= threshold  # Sol bölge
            right_mask = ~left_mask  # Sağ bölge
            
            left_y = y[left_mask]
            right_y = y[right_mask]
            
            # Gini hesapla
            gini = (len(left_y) * gini_index(left_y) + len(right_y) * gini_index(right_y)) / len(y)
            
            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold

# Karar Ağacı Modeli
def decision_tree(X_train, y_train, X_test, max_depth=None):
    def build_tree(X, y, depth=0):
        # Durdurma koşullarının belirlenmesi
        # Düğüm bir yaprak düğüm ise (daha fazla bölünemiyorsa) dur
        # Yada
        # Maksimum derinliğe ulaşıldıysa dur
        if len(np.unique(y)) == 1 or (max_depth and depth >= max_depth):
            return Counter(y).most_common(1)[0][0]  # En çok tekrar eden sınıfı döndür
        
        feature, threshold = best_split(X, y)  # En iyi bölmeyi bul
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        left_tree = build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {'feature': feature, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

    def predict_one(x, tree):
        if isinstance(tree, dict):  # Ağaç dalı
            if x[tree['feature']] <= tree['threshold']:
                return predict_one(x, tree['left'])
            else:
                return predict_one(x, tree['right'])
        else:  # Yaprak
            return tree
    
    # Modeli oluştur
    tree = build_tree(X_train, y_train)
    
    # Test verisi üzerinde tahmini oluştur
    predictions = [predict_one(x, tree) for x in X_test]
    return predictions


#-------------------------------------------------------SVM-------------------------------------------------------
# Hinge loss fonksiyonu ve doğrusal SVM için kayıp fonksiyonu
def hinge_loss(w, b, X, y):
    return np.mean(np.maximum(0, 1 - y * (np.dot(X, w) + b)))

# SVM'yi Sıfırdan Eğitme
def train_svm(X_train, y_train, lr=0.01, epochs=1000, C=1.0):
    # Eğitim için başlangıç parametreleri
    n_features = X_train.shape[1]
    w = np.zeros(n_features)  # Ağırlıklar
    b = 0  # Bias terimi
    
    # Etiketlerin +1 ve -1 olması gerektiği için dönüşüm
    y_train = 2 * y_train - 1  # (0, 1) -> (-1, 1)

    # Eğitim döngüsü
    for _ in range(epochs):
        for i in range(len(X_train)):
            # Veriyi SVM kuralına göre güncelleme
            if y_train[i] * (np.dot(X_train[i], w) + b) < 1:
                # Hata durumunda, ağırlıkları ve bias'ı güncelle
                w = w - lr * (2 * C * w - np.dot(X_train[i], y_train[i]))
                b = b - lr * (2 * C * b - y_train[i])
            else:
                # Doğru sınıflandırma durumunda yalnızca ağırlıkları güncelle
                w = w - lr * 2 * C * w
    
    return w, b

# Modeli Test Etme
def svm_predict(X_test, w, b):
    # Model tahminini yap
    y_pred = ((np.dot(X_test, w) + b) > 0).astype(int) # Eğer tahmin > 0 ise 1, aksi takdirde 0
    return y_pred


#-------------------------------------------------------YAPAY SİNİR AĞLARI-------------------------------------------------------
# Sigmoid Aktivasyon Fonksiyonu ve türevi
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1 - x)

# class SimpleNeuralNetwork:
#     def __init__(self, input_size, hidden_size, output_size):
#         # Ağırlıklar ve biyasları rastgele başlat
#         self.weights_input_hidden = np.random.randn(input_size, hidden_size)  # Girişten gizli katmana
#         self.bias_hidden = np.zeros((1, hidden_size))  # Gizli katman bias
#         self.weights_hidden_output = np.random.randn(hidden_size, output_size)  # Gizli katmandan çıkışa
#         self.bias_output = np.zeros((1, output_size))  # Çıkış katmanı bias

#     def forward(self, X):
#         # İleri yayılım (Forward Propagation)
#         self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
#         self.hidden_output = sigmoid(self.hidden_input)  # Gizli katman çıkışı
#         self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
#         self.final_output = sigmoid(self.final_input)  # Çıkış katmanı çıkışı
#         return self.final_output

#     def backward(self, X, y, learning_rate=0.01):
#         # Geri yayılım (Backpropagation)
#         output_error = y - self.final_output  # Hata (Gerçek - Tahmin)
#         output_delta = output_error * sigmoid_derivative(self.final_output)  # Çıkış katmanındaki delta

#         hidden_error = output_delta.dot(self.weights_hidden_output.T)  # Gizli katmandaki hata
#         hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)  # Gizli katman için delta

#         # Ağırlıkları ve biyasları güncelle
#         self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
#         self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
#         self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
#         self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

#     def train(self, X, y, epochs=1000, learning_rate=0.01):
#         for _ in range(epochs):
#             self.forward(X)  # İleri yayılım
#             self.backward(X, y, learning_rate)  # Geri yayılım