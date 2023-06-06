from haystack.telemetry import tutorial_running
from haystack.document_stores import InMemoryDocumentStore
from haystack import Document
from haystack.nodes.retriever.multimodal import MultiModalRetriever
import os
from haystack import Pipeline




tutorial_running(19)

def getImage(query):
    print(query,'........................')
    document_store = InMemoryDocumentStore(embedding_dim=512)
    doc_dir = 'images/content'
    images = [Document(content=f"{doc_dir}/{filename}", content_type="image") for filename in os.listdir('images/content/') ]
    document_store.write_documents(images)
    print(images)
    retriever_text_to_image = MultiModalRetriever(
    document_store=document_store,
    query_embedding_model="sentence-transformers/clip-ViT-B-32",
    query_type="text",
    document_embedding_models={"image": "sentence-transformers/clip-ViT-B-32"},)

    document_store.update_embeddings(retriever=retriever_text_to_image)
    pipeline = Pipeline()
    pipeline.add_node(component=retriever_text_to_image, name="retriever_text_to_image", inputs=["Query"])
    results = pipeline.run(query=query, params={"retriever_text_to_image": {"top_k": 1}})
    results = sorted(results["documents"], key=lambda d: d.score, reverse=True)

    images_array = [doc.content for doc in results]
    scores = [doc.score for doc in results]

    # print(images_array)
    # print(scores)

    return images_array,scores
#     for ima, score in zip(images_array, scores):
#         im = Image.open(ima)
#         img_with_border = ImageOps.expand(im, border=20, fill="white")

#         img = ImageDraw.Draw(img_with_border)
#         img.text((20, 0), f"Score: {score},    Path: {ima}", fill=(0, 0, 0))

#         bio = BytesIO()
#         img_with_border.save(bio, format="png")   
#         #img1display = display(IPImage(bio.getvalue(), format="png",width = 200))
        
#         return display(IPImage(bio.getvalue(), format="png",width = 200))

    