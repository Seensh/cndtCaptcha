import cv2
import numpy as np
import easyocr

# Caminho para a imagem captcha
caminho_imagem = "C:/CAMINHO/imagem_captcha.png"
caminho_imagem_processada = "C:/CAMINHO/imagem_captcha_processada.png"

# Carregar imagem em escala de cinza
imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

# Redimensionar a imagem para melhorar o OCR
imagem = cv2.resize(imagem, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Aumentar o contraste usando CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
imagem = clahe.apply(imagem)

# Aplicar filtro de desfoque Gaussiano para reduzir ruídos
imagem = cv2.GaussianBlur(imagem, (5, 5), 0)

# Aplicar filtro de borda (Canny) para destacar contornos
imagem = cv2.Canny(imagem, 100, 200)

# Inverter a imagem para OCR
imagem = cv2.bitwise_not(imagem)

# Salvar a imagem processada para verificação
cv2.imwrite(caminho_imagem_processada, imagem)
print(f"Imagem processada salva em: {caminho_imagem_processada}")

# Inicializar o leitor EasyOCR
reader = easyocr.Reader(['en'])

# Realizar OCR na imagem
resultado = reader.readtext(imagem)

# Extrair o texto detectado
texto = ' '.join([res[1] for res in resultado])
print("Texto detectado:", texto)
