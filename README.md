Penelitian ini bertujuan mengembangkan model klasifikasi tutupan lahan vegetasi dari citra resolusi tinggi menggunakan data ISPRS Vaihingen. Model yang diterapkan adalah Convolutional Neural Network (CNN) dengan arsitektur Encoder VGG16-Net. Pengujian melibatkan 8 skenario dengan rasio data latih dan uji 80%:20% serta 70%:30%. Dua metode classifier digunakan: argmax dan threshold, dengan komparasi Neural Network antara 1 dan 2 hidden layer.

Hasil menunjukkan bahwa metode threshold mengurangi waktu training sebesar 44 detik dibandingkan argmax. Penambahan hidden layer pada Neural Network meningkatkan performa model pada metrik recall, akurasi, F1-score, dan IoU, meski sedikit menurunkan presisi. Waktu training optimal adalah 12 menit 36 detik, sedangkan performa terbaik pada lima metrik diperoleh dengan presisi 0.995, recall 0.543, akurasi 0.833, F1-score 0.703, dan IoU 0.542.

<b> Running Program </b>
Untuk menjalankan atau mencoba program klasifikasi piksel atau pixel wise classification (semantic segmentation) ini, bisa dengan menjalankan python app.py pada terminal untuk membuka interface dari Flask.
