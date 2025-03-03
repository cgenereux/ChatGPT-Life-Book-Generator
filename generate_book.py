import json
import os
import glob
from datetime import datetime
import anthropic
import time
from multiprocessing import Process, Manager, Value

class ChatGPTBookGenerator:
    def __init__(self, api_key, input_file, start_idx=0, end_idx=10, output_dir="chapters", worker_id=0):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.input_file = input_file
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.output_dir = output_dir
        self.worker_id = worker_id
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_conversations(self):
        """Load and parse the ChatGPT export JSON file."""
        print(f"Worker {self.worker_id}: Loading conversations from {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def extract_conversation_content(self, conversation):
        """Extract the actual messages from the complex conversation structure."""
        messages = []
        
        # Skip if conversation structure isn't as expected
        if not conversation.get('mapping'):
            return {"title": conversation.get('title', 'Untitled'), "content": []}
        
        # Extract messages from the tree structure
        for msg_id, msg_data in conversation['mapping'].items():
            if not msg_data.get('message'):
                continue
                
            message = msg_data['message']
            if (message.get('content', {}).get('content_type') == 'text' and
                message['content'].get('parts') and
                message['author']['role'] in ['user', 'assistant']):
                
                content = message['content']['parts'][0]
                if content and len(content.strip()) > 0:
                    messages.append({
                        'role': message['author']['role'],
                        'content': content
                    })
        
        return {
            "title": conversation.get('title', 'Untitled'),
            "content": messages
        }
    
    def extract_notable_information(self, conversation):
        """Extract all notable information about the user from a conversation."""
        if not conversation['content']:
            return []
        
        # Format the conversation for the prompt
        formatted_messages = []
        for msg in conversation['content']:
            formatted_messages.append(f"{msg['role'].upper()}: {msg['content']}")
        
        conversation_text = "\n\n".join(formatted_messages)
        
        # Date of conversation (extract from title or use placeholder)
        conv_date = "unknown date"
        if ":" in conversation['title']:
            try:
                conv_date = conversation['title'].split(":")[0].strip()
            except:
                pass
        
        # Use the provided prompt with extra note about ignoring assignments
        prompt = f"""
        CONVERSATION TITLE: {conversation['title']}
        CONVERSATION DATE: {conv_date}
        
        {conversation_text}
        
        Extract notable information about the person from this conversation as bullet points.
        
        Include any information that reveals something about the user's:
        - Opinions or preferences (even minor ones like "user prefers dogs over cats")
        - Facts about their life
        - Interests and activities
        - Relationships or interactions with others
        - Beliefs, values or principles
        - Struggles, concerns or challenges
        - Goals, plans or aspirations
        - Questions they're exploring
        
        Guidelines:
        - Create as many or as few bullet points as appropriate - some conversations might have many insights, others few or none
        - If there's no notable information about the user, just return "No notable information"
        - Keep each bullet point factual and specific
        - Focus on information about the USER, not general knowledge
        - Format as bullet points starting with "-"
        
        Some notes:
        * Ignore overly short-term stuff: No need to document what the user ate on Feb 28, unless it was a life-changing meal.
        * Skip impersonal info: If it's not unique to the user (like general facts about UPS delivery times), it's a waste of tokens.
        * Time-scope insights: Instead of "User feels X about Y," make it time-moment-bounded like "User felt X about Y at a particular point" to reflect changing opinions without sounding absolute.
        * Trim redundant wording: "The user" is costing unnecessary tokens. Claude can just output raw insights in direct, compressed form.
        * Ignore specific assignments or school activities generally. Like particular grunt homework usually isn't noteworthy.
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Using 3.5 for higher RPM
                max_tokens=4000,
                system="Extract only significant, personal, and unique information about the person from their conversations. Focus on durable insights, preferences, and life details. Avoid trivial daily activities or general information. Add time context when available. Use concise, direct phrasing.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            bullets_text = response.content[0].text
            
            if "No notable information" in bullets_text:
                print(f"Worker {self.worker_id}: No notable information in: {conversation['title']}")
                return []
            
            bullets = [bullet.strip() for bullet in bullets_text.split('\n') if bullet.strip().startswith('-')]
            
            print(f"Worker {self.worker_id}: Extracted {len(bullets)} data points from: {conversation['title']}")
            return bullets
        
        except Exception as e:
            error_message = str(e).lower()
            if "rate" in error_message or "limit" in error_message or "overload" in error_message:
                time.sleep(2)
                return []
            else:
                print(f"Worker {self.worker_id}: Error processing conversation '{conversation['title']}': {e}")
                time.sleep(2)
                return []
    
def process_conversation_batch(self, conversations=None):
    if conversations is None:
        conversations = self.load_conversations()
    total_convos = len(conversations)
    print(f"Worker {self.worker_id}: Found {total_convos} total conversations in export file.")
    if self.start_idx >= total_convos:
        print(f"Worker {self.worker_id}: Start index {self.start_idx} exceeds total conversations {total_convos}.")
        return
    end_idx = min(self.end_idx, total_convos)
    batch_convos = conversations[self.start_idx:end_idx]
    print(f"Worker {self.worker_id}: Processing batch from index {self.start_idx} to {end_idx-1} ({len(batch_convos)} conversations)...")
    
    # Extract notable information from each conversation
    all_bullets = []
    for i, convo in enumerate(batch_convos):
        convo_idx = self.start_idx + i
        clean_convo = self.extract_conversation_content(convo)
        print(f"Worker {self.worker_id}: Processing conversation {convo_idx} ({i+1}/{len(batch_convos)}): {clean_convo['title']}")
        
        bullets = self.extract_notable_information(clean_convo)
        all_bullets.extend(bullets)
        
        # No need for delay with Claude's high rate limit, but keep a small one
        # just to avoid overwhelming the API
        time.sleep(0.2)
    
    print(f"Worker {self.worker_id}: Collected {len(all_bullets)} total data points from {len(batch_convos)} conversations.")
    
    # Save the raw bullets
    if all_bullets:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_bullets_file = os.path.join(self.output_dir, f"raw_bullet_points_worker{self.worker_id}_{timestamp}.md")
        with open(raw_bullets_file, 'w', encoding='utf-8') as f:
            f.write(f"# Raw Data Points - Worker {self.worker_id} - Conversations {self.start_idx}-{end_idx-1}\n\n")
            for bullet in all_bullets:
                f.write(f"{bullet}\n")
        
        print(f"Worker {self.worker_id}: Saved {len(all_bullets)} raw bullet points to {raw_bullets_file}")
    
    print(f"Worker {self.worker_id}: Processing complete!")

def run_worker(worker_convos, output_dir, worker_id, api_key):
    generator = ChatGPTBookGenerator(
        api_key=api_key,
        input_file=None,  # Not needed since we're passing conversations directly
        start_idx=0,
        end_idx=len(worker_convos),
        output_dir=output_dir,
        worker_id=worker_id
    )
    generator.process_conversation_batch(conversations=worker_convos)

def combine_results(output_dir, chunk_info=""):
    """Combine results from all workers into a single file, strictly within this directory only."""
    # Find all raw bullet point files in this directory ONLY (not subdirectories)
    raw_files = glob.glob(os.path.join(output_dir, "raw_bullet_points_worker*.md"))
    
    if not raw_files:
        print(f"No worker result files found in {output_dir}.")
        return 0
    
    # Print files being combined for debugging
    print(f"Combining {len(raw_files)} files from {output_dir}:")
    for file in raw_files:
        print(f"  - {os.path.basename(file)}")
    
    # Combine all bullet points
    all_bullets = []
    for file in raw_files:
        with open(file, 'r', encoding='utf-8') as f:
            # Skip the header line
            lines = f.readlines()[2:]
            file_bullets = [line.strip() for line in lines if line.strip()]
            all_bullets.extend(file_bullets)
            print(f"  Added {len(file_bullets)} bullets from {os.path.basename(file)}")
    
    if not all_bullets:
        print("No bullet points found in worker files.")
        return 0
    
    # Save all combined bullet points
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_file = os.path.join(output_dir, f"combined_bullets_{chunk_info}_{timestamp}.md")
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write(f"# Combined Bullet Points {chunk_info}\n\n")
        for bullet in all_bullets:
            f.write(f"{bullet}\n")
    
    print(f"Combined {len(all_bullets)} bullet points from {len(raw_files)} workers into {combined_file}")
    return len(all_bullets)

def get_next_run_number(base_dir):
    """Get the next run number by checking existing Run folders."""
    # Check for existing Run folders
    run_dirs = glob.glob(os.path.join(base_dir, "Run *"))
    
    # Find the highest run number
    highest_run = 0
    for dir in run_dirs:
        try:
            # Extract the run number from the folder name
            run_num = int(os.path.basename(dir).split(" ")[1])
            highest_run = max(highest_run, run_num)
        except (ValueError, IndexError):
            # If the folder doesn't match our naming pattern, skip it
            pass
    
    # Return next run number
    return highest_run + 1

def worker_task(task_queue, output_dir, worker_id, api_key, shared_counter, total_tasks):
    """Worker that pulls tasks from a shared queue and updates a shared counter.
    Only worker 1 prints a progress update (percentage complete and a progress bar) after finishing a task."""
    generator = ChatGPTBookGenerator(
        api_key=api_key,
        input_file=None,  # Not used because we're passing conversations directly.
        start_idx=0,
        end_idx=0,
        output_dir=output_dir,
        worker_id=worker_id
    )
    local_bullets = []
    while True:
        try:
            idx, convo = task_queue.get_nowait()
        except Exception:
            break  # Queue is empty
        # Process the conversation
        bullets = generator.extract_notable_information(generator.extract_conversation_content(convo))
        local_bullets.extend(bullets)
        
        # Update the shared counter for every task completed
        with shared_counter.get_lock():
            shared_counter.value += 1
            current_count = shared_counter.value
        
        # Only worker 1 prints a progress update with a progress bar
        if worker_id == 1:
            percentage = (current_count / total_tasks) * 100
            bar_length = 30
            filled_length = int(round(bar_length * current_count / float(total_tasks)))
            bar = '#' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r{percentage:.2f}% complete [{bar}]", end="", flush=True)
        
        time.sleep(0.2)
        
    # Print a newline when done (so the prompt starts on a new line)
    if worker_id == 1:
        print()
        
    if local_bullets:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_bullets_file = os.path.join(output_dir, f"raw_bullet_points_worker{worker_id}_{timestamp}.md")
        with open(raw_bullets_file, 'w', encoding='utf-8') as f:
            f.write(f"# Raw Data Points - Worker {worker_id}\n\n")
            for bullet in local_bullets:
                f.write(f"{bullet}\n")
        print(f"Worker {worker_id}: Saved {len(local_bullets)} raw bullet points to {raw_bullets_file}")
    print(f"Worker {worker_id}: Processing complete!")

def process_chunk_dynamic(api_key, conversations, chunk_start, chunk_size, chunk_dir, num_workers):
    """Dynamically processes a chunk of conversations using a shared task queue and counter."""
    chunk_end = chunk_start + chunk_size
    chunk_info = f"Chunk_{chunk_start+1}_{chunk_end}"
    print(f"\n{'='*50}")
    print(f"PROCESSING CHUNK: Conversations {chunk_start+1} to {chunk_end}")
    print(f"{'='*50}")
    print(f"Output directory: {chunk_dir}")
    print(f"Using {num_workers} workers dynamically")
    os.makedirs(chunk_dir, exist_ok=True)
    
    manager = Manager()
    task_queue = manager.Queue()
    shared_counter = Value('i', 0)
    total_tasks = chunk_size  # Total tasks in this chunk
    
    # Add tasks (each conversation in this chunk)
    for i in range(chunk_start, chunk_end):
        task_queue.put((i, conversations[i]))
    
    processes = []
    for worker_id in range(num_workers):
        print(f"Deplying worker #{worker_id}")
        p = Process(
            target=worker_task,
            args=(task_queue, chunk_dir, worker_id, api_key, shared_counter, total_tasks)
        )
        processes.append(p)
        p.start()
        time.sleep(1)
    
    # print(f"Waiting for all {len(processes)} workers to complete...")
    for i, p in enumerate(processes):
        p.join()
        print(f"Worker {i} has completed")
    
    print(f"All workers completed, now combining results for {chunk_info}...")
    total_bullets = combine_results(chunk_dir, chunk_info)
    print(f"\nCHUNK COMPLETED: Extracted {total_bullets} bullet points from conversations {chunk_start+1} to {chunk_end}")
    return total_bullets


# Main execution
if __name__ == "__main__":
    API_KEY = input("Enter your API key: ").strip()
    INPUT_FILE = "conversations.json"
    BASE_OUTPUT_DIR = "life_book_chapters"
    CHUNK_SIZE = 50
    NUM_WORKERS = 10
    START_FROM = 0
    
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
    
    run_number = get_next_run_number(BASE_OUTPUT_DIR)
    run_folder = os.path.join(BASE_OUTPUT_DIR, f"Run {run_number}")
    os.makedirs(run_folder, exist_ok=True)
    chunks_folder = os.path.join(run_folder, "Conversation Chunks")
    os.makedirs(chunks_folder, exist_ok=True)
    
    # Log run information...
    
    print(f"Loading conversations from {INPUT_FILE} to determine total count...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    total_conversations = len(conversations)
    total_chunks = (total_conversations - START_FROM + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    print(f"Found {total_conversations} total conversations")
    print(f"Will process from #{START_FROM} in {total_chunks} chunks of {CHUNK_SIZE} each")
    
    current_chunk = 0
    current_start = START_FROM
    total_bullets_collected = 0
    
    chunks_processed_file = os.path.join(run_folder, "processed_chunks.txt")
    with open(chunks_processed_file, 'w') as f:
        f.write("# Processed Chunks\n\n")
    
    while current_start < total_conversations:
        current_chunk += 1
        chunk_end = min(current_start + CHUNK_SIZE, total_conversations)
        chunk_folder = os.path.join(chunks_folder, f"Chunk_{current_start+1}_{chunk_end}")
        os.makedirs(chunk_folder, exist_ok=True)
        
        bullets_in_chunk = process_chunk_dynamic(
            API_KEY, conversations, current_start, chunk_end - current_start, 
            chunk_folder, NUM_WORKERS
        )

        total_bullets_collected += bullets_in_chunk
        
        # Log progress...
        
        current_start = chunk_end
        if current_start >= total_conversations:
            print(f"\nAll chunks completed! Total bullet points collected: {total_bullets_collected}")
            break
        
        continue_input = input("\nPress Enter to continue to the next chunk or type 'quit' to stop: ")
        if continue_input.lower() in ['quit', 'q', 'exit', 'stop']:
            print("Processing stopped by user after completing chunk " + str(current_chunk))
            break
    
    print(f"\nRUN #{run_number} COMPLETED")
    print(f"Processed {current_start - START_FROM} conversations in {current_chunk} chunks")
    print(f"Collected {total_bullets_collected} total bullet points")
    print(f"Results saved to: {run_folder}")