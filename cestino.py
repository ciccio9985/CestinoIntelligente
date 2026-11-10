import time
import cv2
import numpy as np
from gpiozero import OutputDevice
from tflite_runtime.interpreter import Interpreter

MODELLO = "trashnet_x_raspberry.tflite"
ETICHETTE = "etichette.txt"
INDICE_CAMERA = 0 

classi = open(ETICHETTE).read().splitlines()

cam = cv2.VideoCapture(INDICE_CAMERA)

if not cam.isOpened():
    raise SystemExit("Telecamera non trovata.")

#pin 11,13,15,16 del rapberry
motore_sportello = [OutputDevice(p) for p in (17, 27, 22, 23)]

#inventati
pin = {
    "Plastica": (4, 5, 6, 12),
    "Vetro": (13, 16, 17, 18),
    "Carta": (19, 20, 21, 22),
    "Indifferenziata": (23, 24, 25, 26)
}

motori_cestini = {nome: [OutputDevice(p) for p in pins] for nome, pins in pin.items()}

seq = [(1,0,0,0), (1,1,0,0), (0,1,0,0), (0,1,1,0),
       (0,0,1,0), (0,0,1,1), (0,0,0,1), (1,0,0,1)]


interprete = Interpreter(model_path=MODELLO)
interprete.allocate_tensors()
ingresso = interprete.get_input_details()[0]
uscita = interprete.get_output_details()[0]

def ruota_cestino(motore, passi, avanti=True, pausa=0.0001):
    bobine = motori_cestini[motore]
    seq_rotazione = seq if avanti else seq[::-1]
    for i in range(passi):
        for bobina, valore in zip(bobine, seq_rotazione[i % 8]):
            bobina.value = valore
            time.sleep(pausa)
    for bobina in motore:
        bobina.off()

def ruota_sportello(passi, avanti=True, pausa=0.0001):
	seq_rotazione = seq if avanti else seq[::-1]
	print(seq_rotazione)
	for i in range(passi):
		for bobina, valore in zip(motore_sportello, seq[i%8]):
			bobina.value = valore
			time.sleepsleep(pausa)
	for bobina in motore_sportello:
		bobina.off()

def predizione():
    ok, frame = cam.read()
    if not ok:
        raise SystemExit("La telecamera non funziona correttamente")

    img = cv2.resize(frame, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    interprete.set_tensor(ingresso["index"], img[np.newaxis, ...])
    interprete.invoke()
    prob = interprete.get_tensor(uscita["index"])[0]

    i = int(prob.argmax())
    return classi[i], prob[i]

def gradi_in_step(gradi):
    return gradi*4096/360

if __name__ == "__main__":
    while(True):
        #aspetta la chiusura del coperchio/pressione del pulsante
        ris, prob = predizione()
        if(prob < 0.80):
            ris = 4

        if(ris == "Speciale" or ris == "Umido"):
            #caso da gestire
            time.sleep(5)
        
        else:
            ruota_sportello(gradi_in_step(45), avanti=True)
            ruota_cestino(gradi_in_step(30), avanti=True)
            ruota_cestino(gradi_in_step(30), avanti=False)
            ruota_sportello(gradi_in_step(45), avanti=False)


cam.release()
cv2.destroyAllWindows()