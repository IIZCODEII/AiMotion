from utils.emotions import detect_emotions
import unittest

### Tests Unitaires

test_video_path = './test_unitaire'



class PredictionTest(unittest.TestCase):
    """ Test Case utilisé pour tester la fonctionnalité de prediction sur un flux d'image de notre programme.
    Pour ce faire on utilise un video feed pré-établi correspondant a une video en local , dont on connait déja l'émotion exprimée par le sujet et dans des 
    conditions choisis telle que le reseau de neurones devrait trouver le bon résultat. On met donc à la fois en test la capacité d'acquisition des flux et de leur traitements du programme mais également le modèle en lui-meme.
    Ce test est effectué pour les deux possibilité de prédiction : happy/sad. """ 

    def test_detect_emotions_happy(self):
        """ Test de detect_emotions selon le protocole précisé precedemment pour l'emotion happy"""
        self.assertEqual(detect_emotions(Webcam_Usage=False,Video_Source= test_video_path+'/happy.mp4'),'Happy')

    def test_detect_emotions_sad(self):
        """ Test de detect_emotions selon le protocole précisé precedemment pour l'emotion sad"""
        self.assertEqual(detect_emotions(Webcam_Usage=False,Video_Source= test_video_path+'/sad.mp4'),'Sad')    


unittest.main()

