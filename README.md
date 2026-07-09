# 🚀 Lucidia — AI-Powered Image Caption Generator

Lucidia is an AI-powered image caption generator that transforms images into engaging, social media-ready captions. It combines **BLIP Image Captioning** with **Llama 3** to generate accurate image descriptions and convert them into creative Instagram-style captions.

---

## ✨ Features

- 🖼️ Upload images via drag & drop or file picker
- 🤖 AI-generated image descriptions using **BLIP**
- 💬 Instagram-ready captions powered by **Llama 3**
- ☁️ Cloud inference using Hugging Face API
- 💻 Offline caption generation with a local BLIP model
- ⚡ Fast and responsive Next.js frontend
- 🔒 Image validation and secure API handling
- 📱 Clean, modern, and responsive UI

---

## 🛠 Tech Stack

### Frontend
- Next.js 16
- React 19
- TypeScript
- Tailwind CSS v4

### Backend
- FastAPI
- Python
- Uvicorn

### AI Models
- Salesforce BLIP Image Captioning
- Groq Llama 3 (8B)

### APIs
- Hugging Face Inference API
- Groq API

---

## 📂 Project Structure

```text
lucidia/
│
├── backend/
│   ├── main.py
│   └── venv/
│
├── frontend/
│   ├── app/
│   ├── public/
│   ├── package.json
│   └── ...
│
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/<your-username>/lucidia.git
cd lucidia
```

---

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on:

```
http://localhost:3000
```

---

## Backend

```bash
cd backend

python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

Runs on:

```
http://localhost:8000
```

---

## 🔑 Environment Variables

Create a `.env` file inside the backend directory.

```env
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```

---

## 🖼️ Supported Image Formats

- JPEG
- PNG
- WEBP
- GIF
- BMP

Maximum upload size: **8 MB**

---

## 🚀 How It Works

```text
Upload Image
      │
      ▼
Next.js Frontend
      │
      ▼
FastAPI Backend
      │
      ├── Cloud Mode → Hugging Face BLIP
      │
      └── Local Mode → BLIP Model
      │
      ▼
Base Image Caption
      │
      ▼
Groq Llama 3
      │
      ▼
Instagram Caption
      │
      ▼
Display Result
```

---

## 📌 API Endpoint

### Generate Caption

```
POST /api/generate-caption
```

**Request**

- Image File
- `use_cloud` (true / false)

**Response**

```json
{
  "status": "success",
  "data": {
    "base_caption": "A dog running on the beach.",
    "instagram_caption": "Nothing beats chasing sunsets with your best friend 🐶🌅 #BeachLife #GoldenHour #DogLover"
  }
}
```

---

## 📸 Screenshots

> Add screenshots here.

```
/screenshots/home.png
/screenshots/result.png
```

---

## 🔮 Future Improvements

- Multiple caption styles
- Caption regeneration
- Multi-language support
- Dark mode
- Caption history
- Image OCR
- One-click social media sharing

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Open a Pull Request.

---

## 👨‍💻 Author

**Arpit Mishra**

- GitHub: https://github.com/TRiGG3R22
- LinkedIn: https://www.linkedin.com/in/arpit-mishra-8657a5326/

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub!
