# RAG PDF API

Este projeto é uma API desenvolvida com FastAPI que utiliza técnicas de Recuperação de Informação (Retrieval-Augmented Generation - RAG) para responder perguntas baseadas no conteúdo de arquivos PDF. A API carrega um arquivo PDF, processa seu conteúdo e permite que os usuários façam perguntas sobre ele, retornando respostas relevantes e as fontes utilizadas.

## Funcionalidades

- **Carregamento de PDFs**: O código utiliza o `PyPDFLoader` para carregar e processar o conteúdo de arquivos PDF.
- **Divisão de texto**: O texto do PDF é dividido em partes menores usando o `RecursiveCharacterTextSplitter` para facilitar a indexação e recuperação.
- **Criação de vetores**: Os textos processados são transformados em embeddings vetoriais usando o modelo `all-MiniLM-L6-v2` da biblioteca `HuggingFace`.
- **Banco de dados vetorial**: Os embeddings são armazenados em um banco de dados vetorial utilizando o `FAISS`.
- **Modelo de linguagem**: A API utiliza o modelo `llama3-8b-8192` da `ChatGroq` para gerar respostas baseadas nas informações recuperadas.
- **Endpoint `/ask`**: Permite que os usuários façam perguntas e obtenham respostas baseadas no conteúdo do PDF, incluindo as fontes das respostas.

## Como funciona

1. **Inicialização do modelo**:
   - O arquivo PDF `temp.pdf` é carregado.
   - O texto é dividido em partes menores.
   - Os textos são transformados em embeddings e armazenados em um banco de dados vetorial.
   - Um modelo de linguagem é configurado para responder perguntas com base nos dados recuperados.

2. **Endpoint `/ask`**:
   - Recebe uma pergunta no formato JSON.
   - Se a pergunta for `"__init__"`, inicializa o modelo e retorna uma mensagem de boas-vindas.
   - Caso contrário, utiliza o modelo para responder à pergunta e retorna as fontes utilizadas.

## Requisitos

- Python 3.11
- Dependências listadas no arquivo `requirements.txt`:
  - `langchain`
  - `langchain-community`
  - `langchain-groq`
  - `groq`
  - `tavily-python`
  - `pypdf`
  - `faiss-cpu`
  - `python-dotenv`
  - `sentence-transformers`
  - `fastapi`
  - `uvicorn`

## Como executar

1. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
