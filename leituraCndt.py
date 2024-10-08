import cv2
import numpy as np
import pytesseract

# Especifique o caminho completo para o executável do Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Caminho para a imagem captcha
caminho_imagem = "C:/Users/Usuario/Downloads/testecndt/imagem_captcha.png"
caminho_imagem_processada = "C:/Users/Usuario/Downloads/testecndt/imagem_captcha_processada.png"

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

# Realizar OCR com Tesseract
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz'
texto = pytesseract.image_to_string(imagem, config=custom_config)

print("Texto detectado:", texto)
