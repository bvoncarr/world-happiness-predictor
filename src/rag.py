try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

RESOURCES = [
    {
        "id": "gdp_001",
        "category": "economy",
        "title": "GDP and Happiness",
        "content": "Higher GDP per capita is strongly correlated with happiness, but the relationship diminishes at higher income levels. The Easterlin Paradox shows that beyond a certain income threshold, additional wealth contributes less to happiness. Nordic countries like Denmark and Finland maintain high happiness despite not having the world's highest GDP.",
        "source": "World Happiness Report 2024",
    },
    {
        "id": "social_001",
        "category": "social",
        "title": "Social Support and Happiness",
        "content": "Social support — having someone to count on in times of trouble — is one of the strongest predictors of happiness globally. Countries with strong community ties and social safety nets consistently rank higher. Loneliness and social isolation are major drivers of unhappiness across all income levels.",
        "source": "World Happiness Report 2024",
    },
    {
        "id": "health_001",
        "category": "health",
        "title": "Health and Life Expectancy",
        "content": "Healthy life expectancy is a key contributor to happiness scores. Countries investing in public healthcare, nutrition, and disease prevention see consistently higher happiness rankings. Japan and Singapore lead in life expectancy but score lower on happiness due to cultural factors like work-life balance.",
        "source": "WHO Global Health Observatory",
    },
    {
        "id": "freedom_001",
        "category": "freedom",
        "title": "Freedom and Happiness",
        "content": "Freedom to make life choices is a powerful predictor of happiness. This includes political freedom, economic freedom, and personal autonomy. Scandinavian countries score highest because citizens trust institutions and feel empowered to make meaningful life decisions.",
        "source": "World Happiness Report 2024",
    },
    {
        "id": "corruption_001",
        "category": "corruption",
        "title": "Corruption and Trust",
        "content": "Low corruption and high institutional trust are strongly linked to happiness. When citizens trust their government and businesses, they feel more secure and optimistic. Finland and Denmark consistently rank as the least corrupt nations and also the happiest.",
        "source": "Transparency International 2024",
    },
    {
        "id": "generosity_001",
        "category": "generosity",
        "title": "Generosity and Wellbeing",
        "content": "Acts of giving and generosity are linked to personal happiness. Countries with high charitable giving and volunteerism tend to score higher. Research shows that spending money on others increases happiness more than spending on oneself.",
        "source": "Journal of Happiness Studies",
    },
    {
        "id": "nordic_001",
        "category": "country",
        "title": "Why Nordic Countries Are the Happiest",
        "content": "Finland, Denmark, Iceland, Sweden, and Norway consistently top the happiness rankings. Key reasons: strong social safety nets, free education and healthcare, high trust in government, work-life balance, low corruption, and strong community bonds. Finland has ranked #1 for 7 consecutive years.",
        "source": "World Happiness Report 2024",
    },
    {
        "id": "africa_001",
        "category": "country",
        "title": "Happiness Challenges in Sub-Saharan Africa",
        "content": "Sub-Saharan African countries consistently score lowest on happiness due to low GDP, poor healthcare infrastructure, political instability, and low social support. However, some countries like Mauritius and South Africa outperform their economic peers due to strong social ties and community resilience.",
        "source": "World Happiness Report 2024",
    },
    {
        "id": "trend_001",
        "category": "trends",
        "title": "Global Happiness Trends 2005-2025",
        "content": "Global happiness has remained relatively stable over 20 years, but with regional shifts. Eastern Europe and Latin America have seen gains. The COVID-19 pandemic caused a temporary dip in 2020-2021, but recovery was swift in most regions. Young people in wealthy countries are reporting declining happiness since 2012.",
        "source": "World Happiness Report 2025",
    },
    {
        "id": "us_001",
        "category": "country",
        "title": "United States Happiness Decline",
        "content": "The United States has dropped significantly in happiness rankings, falling out of the top 20 for the first time in 2024. Key drivers: rising inequality, declining social trust, political polarization, and increasing loneliness especially among young adults. Despite high GDP, Americans report lower social support than peers.",
        "source": "World Happiness Report 2024",
    },
]


class HappinessRAG:
    COLLECTION_NAME = "happiness_resources"

    def __init__(self, persist_dir=".chromadb"):
        assert HAS_CHROMA, "Run: pip install chromadb sentence-transformers"
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, force=False):
        if self.collection.count() > 0 and not force:
            print(f"Already indexed {self.collection.count()} resources.")
            return
        self.collection.upsert(
            ids=[r["id"] for r in RESOURCES],
            documents=[r["content"] for r in RESOURCES],
            metadatas=[{"category": r["category"], "title": r["title"], "source": r["source"]} for r in RESOURCES],
        )
        print(f"Indexed {len(RESOURCES)} resources.")

    def retrieve(self, query, n_results=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return docs


if __name__ == "__main__":
    rag = HappinessRAG()
    rag.index()
    docs = rag.retrieve("Why is Finland so happy?")
    for doc in docs:
        print(f"\n[{doc['metadata']['title']}]")
        print(doc["content"])
