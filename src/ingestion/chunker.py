from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def chunk_papers(
    parsed_path: str = "./data/parsed_papers.json",
    output_path: str = "./data/chunks.json",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    # load parsed_papers.json
    try:
        with open(parsed_path, 'r') as f:
            parsed_papers = json.load(f)
        print(len(parsed_papers))
    except Exception as e:
        print(f"Error while reading parsed_json file, {e}")
        raise RuntimeError(f"Error while reading parsed_json file, {e}")

    text_splitter = RecursiveCharacterTextSplitter(
            separators = ['\n\n', '\n', '.', ''],
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = len,
            is_separator_regex = False 
        )
    # for each paper, for each page, split text into chunks
    chunks = []
    for paper in parsed_papers:
        for page in paper['pages']:
            try:
                page_chunks = text_splitter.split_text(page['text'])
                for index, chunk in enumerate(page_chunks):
                    # attach metadata (arxiv_id, title, authors, page, chunk_index) to each chunk
                    chunk_json = {
                        'arxiv_id' : paper['arxiv_id'],
                        'title' : paper['title'],
                        'authors' : paper['authors'],
                        'published' : paper['published'],
                        'categories' : paper['categories'],
                        'page' : page['page'],
                        'chunk_index' : index,
                        'chunk_id' : f"{paper['arxiv_id']}_p{page['page']}_c{index}",
                        'chunk_text' : chunk,
                    }
                    chunks.append(chunk_json)
            except Exception as e:
                print(f"error while creating chunks for page {page['page']} for paper {paper['title']}: {e}")

    # save all chunks to output_path
    try:
        with open(output_path, 'w') as f:
            json.dump(chunks, f, indent=2)
    except Exception as e:
        print(f"Error while saving chunks to file, {e}")

            
    print(len(chunks))

    # return list of chunks
    return chunks

