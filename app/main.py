import uvicorn
from fastapi import FastAPI, Query

from src.get_recomendation import get_recommendation

app = FastAPI()


@app.get("/hello")
def read_hello():
    return {"message": "hello world"}


@app.get("/query")
def query_route(query: str = Query(..., description="Search query")):
    results = get_recommendation(query, threshold=.94)
    recommendations = []
    for i, row in results.iterrows():
        recommendations.append(
            {
                "title": f"Document {i}",
                "content": row["OriginalTweet"],
                "relevance": row["Relevance"],
            }
        )
    return {"results": recommendations, "message": "OK"}

def run():
    uvicorn.run("app.main:app", host="127.0.0.1", port=2206, reload=True)


if __name__ == "__main__":
    run()
