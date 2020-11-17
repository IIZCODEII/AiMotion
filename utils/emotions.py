import cv2
import numpy as np
from fastai.vision import ImageDataBunch, get_transforms, imagenet_stats, models, cnn_learner, pil2tensor, Image


def detect_emotions(Webcam_Usage=True,Video_Source='',Enable_Emoji=True):
    
    ###Chargement du modèle de detection des émotions et des données d'architecture associées


    
    '''Paramètre correspondant au chemin d'accès des données permettant dans ce cas de preciser dans l'objet ImageDataBunch
    la structure de notre modele, notamment le nombre de classes d'émotions à distinguer '''
    emotion_model_path = './data'

   
    # Chargement des informations sur l'architecture du modèle
    dta =ImageDataBunch.from_folder(emotion_model_path, ds_tfms=get_transforms(),size=48).normalize(imagenet_stats)
    emotion_labels = dta.classes
    learn = cnn_learner(dta, models.resnet50,pretrained=False)
    # Chargement des poids de notre modèle pré-entrainé sur nos données coresspondant à cette architecture (88% de précision sur la discrimantion Happy/Sad sur l'ensemble de validation)
    learn.load('model-resnet50-1')
    
 
    ### Flux d'images entrant dans le réseau de neurones

    # Chargement de l'algorithme de la cascade de HAAR,technique classique de computer vision,permettant de déjà préselectionner les visages sur le flux d'images entrant

    haar_cascade = cv2.CascadeClassifier('./data/models/haarcascade_frontalface_default.xml')

    # Parametre définissant le rognage effectué sur les visages préselectionnés, permettant de meilleurs performances de detection
    crop_param = (20, 40)

    # Paramètre de chemin d'accès utilisé pour afficher l'émoji correspondant à l'émotion prédite sur le visage : 
    emoji_path = './data/emoji'
    
    # Paramètres utilisés pour effectuer le test unitaire de detect_emotions()
    test_pred = None
    

    # Fenetre de capture
    cv2.namedWindow('AI_MOTION')
    
    # Selection du type de flux, webcam ou vidéo en local
    video_feed_capture = None
    if Webcam_Usage:
        # 0 est par défaut dans CV2 le paramètre corespondant à la webcam
        video_feed_capture = cv2.VideoCapture(0) 
    else:
        # Source Vidéo
        video_feed_capture = cv2.VideoCapture(Video_Source) 

    # Demarrage du flux d'images d'entrant
    while video_feed_capture.isOpened():

        # Extraction de  l'état de lecture et de la frame en cours
        r , frame = video_feed_capture.read()

        # Si la lecture est faites sans encombre, r= True , on peut proceder à la suite du programme
        if r:

            # Par défaut, la frame est en RGB, on la converti en image "niveaux de gris" car c'est ce sur quoi notre modèle a été entrainer
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #Extraction des visages present sur la gray_frame grace à l'algorithme de HAAR précedement chargé, nous retourne un itérable de coordonnées spatiales dans la gray_frame correspondantes
            
            visages = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5,
                    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            # Grace à chaque coordonnées des visages detectés, on leur applique le rognage spécifié precedemment et les envoient dans le réseau de neurones pour detection
            for visage_coordinates in visages:

                # Extraction des coordonées du visage en cours de traitementn, h étant la hauteur et w la largeur du rectangle correspondant au visage
                X, Y, w, h = visage_coordinates

                # Rognage et extraction de l'image 'niveau de gris' traité du visage en cours de traitement
                x_d, y_d = crop_param
                x1, x2, y1, y2 = (X - x_d, X + w + x_d, Y - y_d, Y + h + y_d)
                cp_visage = gray_frame[y1:y2, x1:x2]

                # Conversion de l'image dans une représentation non biasée par cv2.COLOR_BGR2GRAY et PIL et compréhensible par pytorch et donc fastai 
                
                cp_visage = Image(pil2tensor(cp_visage,np.float32).div_(255))


                # Appel du modèle chargé pour obtenir sa prédiction en terme de classe d'émotion ainsi que sa confiance sur cette prédiction en terme de probabilité
                emotion_pred_class, pred_idx, outputs = learn.predict(cp_visage)
                emotion_prob = float(outputs[np.argmax(outputs)])

                # Assignation de la prediction faite a une variabLE nous permettant d'éffectuer un test unitaire
                test_pred = str(emotion_pred_class)
                
                # Affichage,sur la frame RGB d'origine, du rectangle issu de la HAAR cascade sur le visage detecté, ainsi que sur celui-ci l'émotion prédite
                cv2.rectangle(img=frame,pt1=(X, Y),pt2=(X + w, Y + h),color=(0, 0, 0),thickness=2)
                cv2.putText(frame,str(emotion_pred_class),(X, Y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0, 128, 0),2)
                # Affichage, en bas à gauche de la frame, de la confiance que le modèle a en sa prédiction 
                cv2.putText(frame,str(emotion_pred_class) + ' Probability  ' + str(round(emotion_prob,3)*100)+"%",(50,600),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),3)

                
                #Affichage, sur une fenetre annexe, de l'émoji correspondant à l'émotion prédite sur le visage en traitement , dans le cas ou Enable_Emoji est vrai et que la webcam est utilisée
                
                if (Enable_Emoji and Webcam_Usage) :

                    if str(emotion_pred_class) == 'Happy':
                        
                        emoji = cv2.imread(emoji_path+'/happy.png',1)
                        cv2.namedWindow('emoji', cv2.WINDOW_NORMAL)
                        cv2.imshow('emoji',emoji)
                    
                    else:
                        
                        emoji = cv2.imread(emoji_path+'/sad.png',1)
                        cv2.namedWindow('emoji', cv2.WINDOW_NORMAL)
                        cv2.imshow('emoji',emoji)
                    
                    
            # Affichage de tous les traitements effectués,à chaque nouvelle frame, sur la frame originelle dans la fenetre principale 'AI_MOTION'    
            cv2.imshow('AI_MOTION', frame)
            # Definition d'une escape key permetant de sortir de la boucle crop-detection-affichage
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #Si l'état de lecture retourne False, on sort de la boucle while d'ouverture de la capture du flux d'image        
        else:
            break        

    # Fin de la capture du flux d'image et fermeture des fenetres emoji et AI_MOTION       
    video_feed_capture.release()
    cv2.destroyAllWindows()
    # La fonction retourne la valeur de test_pred, correspondant à la prédiction sur la dernière frame du dernier visage, a des fins de tests
    return test_pred



detect_emotions()







