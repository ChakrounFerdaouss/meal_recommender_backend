from .service import CalorieDLService
 
# Singleton — l'entraînement est lazy (premier appel à .estimate())
service = CalorieDLService()