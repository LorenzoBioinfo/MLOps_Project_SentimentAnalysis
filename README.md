---
title: "SentimentAnalysis"
colorFrom: "red"
colorTo: "green"
sdk: "docker"
sdk_version: "{{1.0}}"
app_file: src/app.py
pinned: false

---
![CI](https://github.com/LorenzoBioinfo/MLOps_Project_SentimentAnalysis/actions/workflows/ci.yml/badge.svg)

# 🧠 Sentiment Analysis MLOps Pipeline  
**MachineInnovators Inc.** — Scalable Machine Learning for Social Reputation Monitoring  

## 📘 Descrizione  
Progetto dedicato all’automazione dell’**analisi del sentiment** sui social media tramite **FastText** e pratiche **MLOps**.  
L’obiettivo è permettere a MachineInnovators Inc. di monitorare la reputazione online, reagire ai cambiamenti nel sentiment degli utenti e mantenere il modello aggiornato nel tempo.  

---

## 🚀 Obiettivi
- **Analisi automatica del sentiment** (positivo / neutro / negativo)  
- **Monitoraggio continuo** delle performance e del sentiment nel tempo  
- **Retraining automatico** del modello per adattarsi ai nuovi dati  

---

## 🧩 Struttura del Progetto
### **Fase 1 — Modello**
- Modello: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)  
- Dataset pubblici con testi etichettati per il sentiment  

### **Fase 2 — Pipeline CI/CD**
- Automazione di training, test e deploy tramite **GitHub Actions** e **Docker**

### **Fase 3 — Deploy e Monitoraggio** 
- Deploy su **Hugging Face**
- Monitoraggio continuo 

---

## 🛠️ Stack Tecnologico
| Categoria | Tecnologie |
|------------|-------------|
| Linguaggio | Python |
| ML | FastText, Transformers, Scikit-learn |
| CI/CD | GitHub Actions, Docker |
| Deploy | Hugging Face Hub |


---


## 💡 Motivazione
L’analisi automatica del sentiment consente una gestione più efficiente e reattiva della reputazione aziendale.  
Con questo progetto, MachineInnovators Inc. integra soluzioni MLOps per una pipeline scalabile, affidabile e costantemente aggiornata.  


## ⚙️ Spiegazione del Progetto

Il progetto è organizzato in tre moduli principali:

### **1️⃣ Applicazione di Sentiment Analysis (FastAPI App)**
L’applicazione, sviluppata in **FastAPI**, offre tre endpoint di analisi:

- **/random_tweet** – analizza esempi tratti dal dataset *TweetEval*  
- **/random_youtube** – valuta esempi provenienti dal dataset di *commenti YouTube*  
- **/predict** – consente all’utente di inserire manualmente una frase per l’analisi del sentiment  

Il modello utilizzato è **[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)**, ottimizzato per la classificazione in tre classi di sentiment.

---

### **2️⃣ Pipeline di Addestramento e Monitoraggio**
L’intero processo di addestramento e valutazione è automatizzato.

- I dati (*TweetEval* e *YouTube Comments*) vengono preprocessati e salvati in formato `datasets.Dataset`.
- Il modello viene valutato con uno script di *monitoring* che calcola **Accuracy** e **F1-score** su entrambi i dataset.
- Se le metriche ottenute sul dataset di YouTube risultano **inferiori a una soglia predefinita** (es. `Accuracy < 0.75`), il sistema esegue automaticamente un **retraining incrementale**, combinando i dati di *TweetEval* con un campione di *YouTube Comments*.



---

### **3️⃣ Pipeline CI/CD e Deploy su Hugging Face**
La pipeline CI/CD, implementata con **GitHub Actions**, automatizza:

1. **Installazione e test del progetto**  
   - Esecuzione di unit test e integration test.  
   - Verifica delle metriche e della qualità del codice.

2. **Retraining automatico**  
   - Se le performance scendono sotto soglia, viene eseguito un retraining parziale.

3. **Deploy e sincronizzazione automatica**  
   - Il modello aggiornato viene pubblicato su **Hugging Face Hub**.  
   - L’app FastAPI viene automaticamente **distribuita come Space**, sempre sincronizzata con l’ultima versione del modello.

---



---

