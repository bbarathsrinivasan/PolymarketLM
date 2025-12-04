# Why We Call It RAG (Retrieval-Augmented Generation)

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that enhances LLM responses by retrieving relevant information from external sources before generating a response. The general architecture is:

1. **Retrieval**: Fetch relevant information from external sources
2. **Augmentation**: Add this information to the prompt
3. **Generation**: Generate a response using the augmented prompt

## Why "RAG" Even Though We Only Use Search?

### RAG is a General Concept

RAG doesn't specify *which* retrieval method to use. Common retrieval sources include:

- **Web Search** (what we're using) - DuckDuckGo, Google, Bing
- **News Feeds** - RSS feeds, news APIs
- **Knowledge Bases** - Wikipedia, domain-specific databases
- **Document Stores** - Vector databases, document embeddings
- **APIs** - Real-time data from various services

### Our Implementation

In this project, we use **web search** as our retrieval method:

- **Retrieval**: Search the web using DuckDuckGo for relevant news/articles
- **Augmentation**: Add search results to the prompt
- **Generation**: Model generates response using both original context and search results

This is still RAG because:
1. ✅ We retrieve external information (web search results)
2. ✅ We augment the prompt with this information
3. ✅ We generate responses using the augmented context

### Why Not Just Call It "Search-Augmented"?

While "search-augmented" would be more specific, "RAG" is:
- The standard term in the research community
- More general (allows for future expansion to other retrieval methods)
- Recognized terminology that researchers understand

### Could We Use Other Retrieval Methods?

Yes! The RAG framework allows for multiple retrieval sources:

1. **News Feeds** (already implemented in `integration/news_retriever.py`)
   - RSS feeds from Reuters, Bloomberg, etc.
   - More structured but may miss breaking news

2. **Hybrid Approach** (search + news)
   - Combine web search and RSS feeds
   - More comprehensive coverage

3. **Vector Databases**
   - Embed documents and retrieve similar ones
   - Better for domain-specific knowledge

4. **APIs**
   - Real-time financial data
   - Weather, sports scores, etc.

### Current Implementation Details

Our RAG pipeline:
```
Input Question → Extract Market Question → Web Search (DuckDuckGo) 
→ Retrieve Top 5 Results → Augment Prompt → Generate Response with Citations
```

This is a valid RAG implementation, even though we're only using one retrieval method (web search).

## Summary

- **RAG** = Retrieval-Augmented Generation (general technique)
- **Our RAG** = Web search-based retrieval + prompt augmentation + generation
- We call it RAG because it follows the RAG paradigm, even with a single retrieval source
- The term is accurate and allows for future expansion to other retrieval methods

