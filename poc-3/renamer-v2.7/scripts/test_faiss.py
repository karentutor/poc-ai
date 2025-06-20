import numpy as np
import faiss

try:
    print(f"FAISS version: {faiss.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Create a small test index
    d = 64                           # dimension
    nb = 100                         # database size
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    
    index = faiss.IndexFlatL2(d)     # build the index
    print(f"Index trained: {index.is_trained}")
    index.add(xb)                    # add vectors to the index
    print(f"Index size: {index.ntotal}")
    
    k = 4                            # we want to see 4 nearest neighbors
    nq = 1                           # let's query 1 vector
    xq = np.random.random((nq, d)).astype('float32')
    
    distances, indices = index.search(xq, k)  # search
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")
    
    print("✅ FAISS import and test successful!")
except Exception as e:
    print(f"❌ FAISS test failed: {str(e)}")
