from multiprocessing import Process, Manager, Value
from datetime import datetime
import anthropic
import threading
import shutil
import json
import glob
import time
import os
import re

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
            
            bullets = [bullet.strip() for bullet in bullets_text.split('\n') if bullet.strip().startswith('-')]
            
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
            
            # print(f"Worker {self.worker_id}: Saved {len(all_bullets)} raw bullet points to {raw_bullets_file}")

def worker_task(task_queue, output_dir, worker_id, api_key, shared_counter, total_tasks, print_lock):
    """Worker that pulls tasks from a shared queue and updates a shared counter."""
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
        clean_convo = generator.extract_conversation_content(convo)
        bullets = generator.extract_notable_information(clean_convo)
        if bullets:
            local_bullets.extend(bullets)
        
        # Update the shared counter for every task completed
        with shared_counter.get_lock():
            shared_counter.value += 1
        
        # Small delay to prevent API rate limits
        time.sleep(0.2)
    
    # Save the bullets collected by this worker
    if local_bullets:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_bullets_file = os.path.join(output_dir, f"raw_bullet_points_worker{worker_id}_{timestamp}.md")
        try:
            with open(raw_bullets_file, 'w', encoding='utf-8') as f:
                f.write(f"# Raw Data Points - Worker {worker_id}\n\n")
                for bullet in local_bullets:
                    f.write(f"{bullet}\n")
            #print(f"Worker {worker_id} saved {len(local_bullets)} bullets")
        except Exception as e:
            print(f"ERROR: Worker {worker_id} failed to save file: {e}")

def combine_results(output_dir, chunk_info=""):
    """Combine results from all workers into a single file."""
    # Find all raw bullet point files in this directory
    raw_files = glob.glob(os.path.join(output_dir, "raw_bullet_points_worker*.md"))
    
    if not raw_files:
        print(f"No worker result files found in {output_dir}.")
        return 0
    
    print(f"Combining {len(raw_files)} worker files...")
    
    # Combine all bullet points
    all_bullets = []
    for file in raw_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                # Skip the header line
                lines = f.readlines()[2:]
                file_bullets = [line.strip() for line in lines if line.strip()]
                all_bullets.extend(file_bullets)
                # print(f"  Added {len(file_bullets)} bullets from {os.path.basename(file)}")
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    
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
    
    return len(all_bullets)

def get_next_book_number(base_dir):
    """Get the next book number by checking existing Book folders."""
    # Check for existing Book folders
    book_dirs = glob.glob(os.path.join(base_dir, "Book-*"))
    
    # Find the highest book number
    highest_book = 0
    for dir in book_dirs:
        try:
            # Extract the book number from the folder name
            book_num = int(os.path.basename(dir).split("-")[1])
            highest_book = max(highest_book, book_num)
        except (ValueError, IndexError):
            # If the folder doesn't match our naming pattern, skip it
            pass
    
    # Return next book number
    return highest_book + 1

def process_chunk_dynamic(api_key, conversations, chunk_start, chunk_size, chunk_dir, num_workers, book_number):
    """Processes conversations using a shared task queue and progress monitor."""
    # Start timing the process
    start_time = time.time()
    
    chunk_end = min(chunk_start + chunk_size, len(conversations))
    actual_size = chunk_end - chunk_start
    chunk_info = f"Book_{book_number}"

    print(f"\n{'='*50}")
    print(f"Generating a book from {actual_size} of {len(conversations)} conversations")
    print(f"{'='*50}\n")
    print(f"Deploying {num_workers} Claude agents...")
    os.makedirs(chunk_dir, exist_ok=True)
    
    manager = Manager()
    task_queue = manager.Queue()
    shared_counter = Value('i', 0)
    print_lock = manager.Lock()
    total_tasks = actual_size

    # Add tasks (each conversation in this chunk) to the queue
    for i in range(chunk_start, chunk_end):
        task_queue.put((i, conversations[i]))
    
    # Create a dedicated function for the progress bar with time estimates
    def progress_monitor():
        last_count = 0
        progress_start_time = time.time()
        
        while True:
            with shared_counter.get_lock():
                current = shared_counter.value
            
            # Only update when the count changes
            if current != last_count:
                # Calculate progress
                percentage = (current / total_tasks) * 100
                bar_length = 30
                filled_length = int(bar_length * current / float(total_tasks))
                bar = '#' * filled_length + '-' * (bar_length - filled_length)
                
                # Calculate time estimates
                elapsed_time = time.time() - progress_start_time
                if current > 0:
                    time_per_task = elapsed_time / current
                    remaining_tasks = total_tasks - current
                    est_remaining_time = remaining_tasks * time_per_task
                    
                    # Format the estimated time
                    if est_remaining_time < 60:
                        time_str = f"{est_remaining_time:.0f} seconds"
                    elif est_remaining_time < 3600:
                        time_str = f"{est_remaining_time/60:.1f} minutes"
                    else:
                        time_str = f"{est_remaining_time/3600:.1f} hours"
                    
                    with print_lock:
                        print(f"\r{percentage:.2f}% complete [{bar}] - Est. remaining: {time_str}", end="", flush=True)
                else:
                    with print_lock:
                        print(f"\r{percentage:.2f}% complete [{bar}] - Calculating time estimate...", end="", flush=True)
                
                last_count = current
            
            # Exit when all tasks are complete
            if current >= total_tasks:
                with print_lock:
                    print()  # Print a final newline
                break
                
            time.sleep(0.5)  # Check every half second
    
    # Start the progress monitor thread before starting workers
    progress_thread = threading.Thread(target=progress_monitor)
    progress_thread.daemon = True  # This ensures the thread will exit when the main program exits
    progress_thread.start()
    
    # Start workers
    processes = []
    for worker_id in range(num_workers):
        p = Process(
            target=worker_task,
            args=(task_queue, chunk_dir, worker_id, api_key, shared_counter, total_tasks, print_lock)
        )
        processes.append(p)
        p.start()
        time.sleep(0.5)  # Stagger worker starts
    
    # Wait for all workers to complete
    for p in processes:
        p.join()
    
    # Make sure progress thread finishes
    if progress_thread.is_alive():
        progress_thread.join(timeout=2)
    
    # Calculate total time taken
    end_time = time.time()
    total_time = end_time - start_time
    
    # Format the total time
    if total_time < 60:
        time_taken_str = f"{total_time:.1f} seconds"
    elif total_time < 3600:
        time_taken_str = f"{total_time/60:.1f} minutes"
    else:
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        time_taken_str = f"{hours} hours, {minutes} minutes"
    
    # Combine and report results
    total_bullets = combine_results(chunk_dir, chunk_info)
    print(f"\n{'='*50}")
    print(f"Book completed in {time_taken_str} with {total_bullets} bullet points.")
    print(f"{'='*50}\n")
    return total_bullets

def create_book(api_key, num_convos, base_output_dir, input_file):
    """Creates a new book using the specified number of conversations."""
    # Create output directories
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    book_number = get_next_book_number(base_output_dir)
    book_folder = os.path.join(base_output_dir, f"Book-{book_number}")
    os.makedirs(book_folder, exist_ok=True)
    results_folder = os.path.join(book_folder, "Results")
    os.makedirs(results_folder, exist_ok=True)

    # Load conversations
    with open(input_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    total_available = len(conversations)
    if num_convos > total_available:
        print(f"Only {total_available} conversations available. Using all.")
        num_convos = total_available

    # Determine how many workers to use (adjust based on available conversations)
    num_workers = min(10, num_convos)
    
    # Process the conversations
    total_bullets_collected = process_chunk_dynamic(
        api_key, conversations, 0, num_convos,
        results_folder, num_workers, book_number
    )
    
    # Find the combined bullets file and copy it to the book folder
    combined_files = glob.glob(os.path.join(results_folder, "combined_bullets_*.md"))
    book_created = False

    if combined_files:
        newest_file = max(combined_files, key=os.path.getctime)
        book_path = os.path.join(book_folder, f"ChatGPT Life Book #{book_number}.md")
        
        # Get first and last conversation titles for metadata
        first_convo_title = conversations[0].get('title', 'Untitled')
        last_convo_title = conversations[num_convos-1].get('title', 'Untitled')
        
        # Copy the combined file as the final book with minimal formatting
        with open(newest_file, 'r', encoding='utf-8') as source:
            with open(book_path, 'w', encoding='utf-8') as dest:
                dest.write(f"# ChatGPT Life Book #{book_number}\n\n")
                dest.write(f"<!-- Book Metadata: first_convo=\"{first_convo_title}\" last_convo=\"{last_convo_title}\" -->\n\n")
                # Skip the header lines (first two lines) from the source
                lines = source.readlines()[2:]
                for line in lines:
                    dest.write(f"{line.strip()}\n")
        
        # Save a metadata file with conversation titles
        metadata_path = os.path.join(book_folder, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "first_conversation": first_convo_title,
                "last_conversation": last_convo_title,
                "creation_date": datetime.now().isoformat(),
                "bullets": total_bullets_collected
            }, f)

        book_created = True
        print(f"Created book with {total_bullets_collected} bullet points.")
        print(f"Book saved to: {book_path}")
    else:
        print("No combined bullet file was found. Book creation failed.")

    # Delete the temporary results folder
    if os.path.exists(results_folder):
        try:
            shutil.rmtree(results_folder)
            # print("Cleaned up temporary files.")
        except Exception as e:
            print(f"Error removing results folder: {e}")

    return book_created, book_number

def index_convo(input_file, title_to_find=None):
    """If title_to_find is provided, prints the index of the matching conversation.
    Otherwise, lists all conversation indices and titles."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
    except Exception as e:
        print(f"Error loading conversations: {e}")
        return

    if title_to_find:
        for idx, convo in enumerate(conversations):
            title = convo.get('title', 'Untitled')
            if title.lower() == title_to_find.lower():
                print(f"Conversation '{title_to_find}' found at index {idx}")
                return idx
        print(f"Conversation '{title_to_find}' not found.")
    else:
        print("\n=== Conversation Index ===")
        for idx, convo in enumerate(conversations):
            title = convo.get('title', 'Untitled')
            print(f"{idx}: {title}")
        print(f"Total Conversations: {len(conversations)}\n")

def expand_book(api_key, book_number, before_convos, after_convos, base_output_dir, input_file):
    """
    Expands an existing book by simply prepending new (newer) bullet points
    and/or appending older bullet points without splicing the original content.
    - before_convos: How many older conversations to add (at the end)
    - after_convos: How many newer conversations to add (at the beginning)
    Note: Conversations are in reverse chronological order (index 0 = newest)
    """
    import os, json, glob, re, shutil, traceback
    from datetime import datetime
    
    try:
        print(f"Starting expand_book for Book #{book_number}")
        print(f"Parameters: before_convos={before_convos}, after_convos={after_convos}")
        
        # Locate book folder.
        book_folder = os.path.join(base_output_dir, f"Book-{book_number}")
        print(f"Looking for book folder at {book_folder}")
        if not os.path.exists(book_folder):
            print(f"Book #{book_number} not found.")
            return False

        # Load metadata (or extract from a book file).
        metadata_path = os.path.join(book_folder, "metadata.json")
        print(f"Metadata path: {metadata_path}")
        
        current_first, current_last = None, None
        if os.path.exists(metadata_path):
            print("Found metadata.json, loading...")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            current_first = metadata.get("first_conversation")
            current_last = metadata.get("last_conversation")
            print(f"From metadata.json: first_conversation={current_first}, last_conversation={current_last}")
        else:
            print("No metadata.json found, looking for metadata in book files...")
            book_files = glob.glob(os.path.join(book_folder, "*.md"))
            if not book_files:
                print(f"No book file found in {book_folder}.")
                return False
            
            book_file = max(book_files, key=os.path.getctime)
            print(f"Using book file: {book_file}")
            
            with open(book_file, 'r') as f:
                for line in f:
                    if "<!-- Book Metadata:" in line:
                        print(f"Found metadata line: {line.strip()}")
                        first_match = re.search(r'first_convo="([^"]+)"', line)
                        last_match = re.search(r'last_convo="([^"]+)"', line)
                        if first_match and last_match:
                            current_first = first_match.group(1)
                            current_last = last_match.group(1)
                            print(f"From book file: first_convo={current_first}, last_convo={current_last}")
                            break
        
        if not current_first or not current_last:
            print("Incomplete metadata. Cannot expand this book.")
            return False

        print(f"Current book boundaries - newest: \"{current_first}\", oldest: \"{current_last}\"")
        
        # Calculate file size to estimate conversations
        file_size = os.path.getsize(input_file)
        print(f"JSON file size: {file_size/1024/1024:.2f} MB")
        
        # Use a different approach for large files - count conversations more accurately
        print("Counting total conversations in JSON file...")
        total_convos = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Count occurrences of "title" - each conversation has one
                # This is a quick approximation
                total_convos += line.count('"title":')
        
        print(f"Estimated total conversations: {total_convos}")
        print("Note: This is an approximation based on the file structure")
        
        # For extremely large files, we need a smarter approach to find just the conversations we need
        # First, preprocess to find the book's current boundaries
        first_idx = None  # newest conversation
        last_idx = None   # oldest conversation
        
        # Create a temporary file to track our findings
        temp_dir = os.path.join(book_folder, "Temp")
        os.makedirs(temp_dir, exist_ok=True)
        boundaries_file = os.path.join(temp_dir, "conversation_boundaries.txt")
        
        # First pass - Find just the book boundaries and calculate total conversations
        # Use a more robust approach
        print("Searching for book boundaries in JSON file...")
        idx = 0
        search_complete = False
        
        # Use a line-by-line approach for better memory efficiency
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                in_object = False
                current_object = ""
                object_number = 0
                
                for line in f:
                    # Track if we're inside a conversation object
                    if "{" in line and not in_object:
                        in_object = True
                        current_object = line
                    elif in_object:
                        current_object += line
                        
                        # Check if this is a complete object
                        if "}" in line:
                            # Count nested braces to make sure this is actually the end of the object
                            open_count = current_object.count("{")
                            close_count = current_object.count("}")
                            
                            if open_count == close_count:
                                # We have a complete object
                                in_object = False
                                
                                # Extract the title
                                title_match = re.search(r'"title"\s*:\s*"([^"]+)"', current_object)
                                if title_match:
                                    title = title_match.group(1)
                                    
                                    # Check if this is one of our boundary conversations
                                    if title == current_first:
                                        first_idx = object_number
                                        print(f"Found newest conversation at index {object_number}: \"{title}\"")
                                    
                                    if title == current_last:
                                        last_idx = object_number
                                        print(f"Found oldest conversation at index {object_number}: \"{title}\"")
                                    
                                    # If we found both, we can record them and continue searching
                                    if first_idx is not None and last_idx is not None and not search_complete:
                                        search_complete = True
                                        with open(boundaries_file, 'w') as bf:
                                            bf.write(f"{first_idx},{last_idx},{object_number}")
                                
                                object_number += 1
                                current_object = ""
                                
                                # Print progress for large files
                                if object_number % 1000 == 0:
                                    print(f"Processed {object_number} conversations...")
            
            # Save the final count if we haven't found both boundaries
            if not search_complete:
                with open(boundaries_file, 'w') as bf:
                    bf.write(f"{first_idx},{last_idx},{object_number}")
                    
            total_objects = object_number
            print(f"Finished scanning. Found {total_objects} total conversations.")
            
        except Exception as e:
            print(f"Error scanning JSON file: {e}")
            traceback.print_exc()
            return False
            
        # Check if we found the boundary conversations
        if first_idx is None:
            print(f"Could not find newest conversation \"{current_first}\" in the JSON file.")
            return False
        
        if last_idx is None:
            print(f"Could not find oldest conversation \"{current_last}\" in the JSON file.")
            return False
            
        print(f"Current book contains conversations from index {first_idx} (newest) to {last_idx} (oldest)")
        print(f"Total conversations in file: {total_objects}")
        
        # Calculate boundaries for expansion
        new_first_idx = first_idx
        if after_convos > 0 and first_idx > 0:
            new_first_idx = max(0, first_idx - after_convos)
            newer_count = first_idx - new_first_idx
            if newer_count > 0:
                print(f"Will include {newer_count} newer conversations (indices {new_first_idx} to {first_idx-1})")
            else:
                print("No newer conversations available (already at the beginning of the file)")
        
        new_last_idx = last_idx
        if before_convos > 0:
            new_last_idx = min(total_objects - 1, last_idx + before_convos)
            older_count = new_last_idx - last_idx
            if older_count > 0:
                print(f"Will include {older_count} older conversations (indices {last_idx+1} to {new_last_idx})")
            else:
                print("No older conversations available (already at the end of the file)")
        
        # Check if there's anything to expand
        if new_first_idx == first_idx and new_last_idx == last_idx:
            print("No new conversations to add - book already contains the requested range.")
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False

        # Create temporary results folder.
        results_folder = os.path.join(book_folder, "Results")
        os.makedirs(results_folder, exist_ok=True)
        print(f"Created temporary results folder at {results_folder}")
        
        # Process newer conversations if requested and available
        new_newer_bullets = ""
        if after_convos > 0 and new_first_idx < first_idx:
            num_newer = first_idx - new_first_idx
            print(f"Processing {num_newer} newer conversations...")
            
            # Extract newer conversations line by line
            newer_convos = []
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    in_object = False
                    current_object = ""
                    object_number = 0
                    
                    for line in f:
                        # Track if we're inside a conversation object
                        if "{" in line and not in_object:
                            in_object = True
                            current_object = line
                        elif in_object:
                            current_object += line
                            
                            # Check if this is a complete object
                            if "}" in line:
                                # Count nested braces to ensure this is the end
                                open_count = current_object.count("{")
                                close_count = current_object.count("}")
                                
                                if open_count == close_count:
                                    # Complete object
                                    in_object = False
                                    
                                    # If this is in our newer range, keep it
                                    if new_first_idx <= object_number < first_idx:
                                        try:
                                            convo_obj = json.loads(current_object)
                                            newer_convos.append(convo_obj)
                                            title = convo_obj.get('title', 'Untitled')
                                            print(f"  Extracted newer conversation {object_number}: \"{title}\"")
                                        except json.JSONDecodeError:
                                            print(f"  Error parsing conversation {object_number}")
                                    
                                    object_number += 1
                                    
                                    # If we've gone past the range we need, we can stop
                                    if object_number >= first_idx:
                                        break
                                        
                                    current_object = ""
            except Exception as e:
                print(f"Error extracting newer conversations: {e}")
                traceback.print_exc()
            
            if newer_convos:
                print(f"Successfully extracted {len(newer_convos)} newer conversations")
                
                # Process the newer conversations
                num_workers = min(len(newer_convos), 10)
                if before_convos > 0:
                    num_workers = min(num_workers, 5)
                
                print(f"Calling process_chunk_dynamic with num_workers={num_workers}...")
                try:
                    process_chunk_dynamic(api_key, newer_convos, 0, len(newer_convos),
                                        results_folder, num_workers, book_number)
                    print("process_chunk_dynamic completed successfully for newer conversations")
                except Exception as e:
                    print(f"Error in process_chunk_dynamic for newer conversations: {e}")
                    traceback.print_exc()
                
                combined_files = glob.glob(os.path.join(results_folder, "combined_bullets_*.md"))
                print(f"Found {len(combined_files)} combined files after processing newer conversations")
                
                if combined_files:
                    newer_file = max(combined_files, key=os.path.getctime)
                    newer_target = os.path.join(results_folder, "newer_bullets.md")
                    print(f"Renaming {newer_file} to {newer_target}")
                    os.rename(newer_file, newer_target)
                    
                    with open(newer_target, 'r') as f:
                        lines = f.readlines()
                    
                    # Skip header lines if they exist
                    start_idx = 0
                    for i, line in enumerate(lines):
                        if line.strip() == "":
                            start_idx = i + 1
                            break
                    
                    new_newer_bullets = "".join(lines[start_idx:])
                    print(f"Extracted {len(new_newer_bullets.splitlines())} new bullet points from newer conversations")
            else:
                print("No newer conversations were successfully extracted.")
        else:
            print("No newer conversations requested or available for expansion.")

        # Process older conversations if requested and available
        new_older_bullets = ""
        if before_convos > 0 and new_last_idx > last_idx:
            num_older = new_last_idx - last_idx
            print(f"Processing {num_older} older conversations...")
            
            # Extract older conversations line by line
            older_convos = []
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    in_object = False
                    current_object = ""
                    object_number = 0
                    
                    for line in f:
                        # Track if we're inside a conversation object
                        if "{" in line and not in_object:
                            in_object = True
                            current_object = line
                        elif in_object:
                            current_object += line
                            
                            # Check if this is a complete object
                            if "}" in line:
                                # Count nested braces to ensure this is the end
                                open_count = current_object.count("{")
                                close_count = current_object.count("}")
                                
                                if open_count == close_count:
                                    # Complete object
                                    in_object = False
                                    
                                    # If this is in our older range, keep it
                                    if last_idx < object_number <= new_last_idx:
                                        try:
                                            convo_obj = json.loads(current_object)
                                            older_convos.append(convo_obj)
                                            title = convo_obj.get('title', 'Untitled')
                                            print(f"  Extracted older conversation {object_number}: \"{title}\"")
                                        except json.JSONDecodeError:
                                            print(f"  Error parsing conversation {object_number}")
                                    
                                    object_number += 1
                                    
                                    # If we've gone past the range we need, we can stop
                                    if object_number > new_last_idx:
                                        break
                                        
                                    current_object = ""
            except Exception as e:
                print(f"Error extracting older conversations: {e}")
                traceback.print_exc()
            
            if older_convos:
                print(f"Successfully extracted {len(older_convos)} older conversations")
                
                # Process the older conversations
                num_workers = min(len(older_convos), 10)
                if after_convos > 0:
                    num_workers = min(num_workers, 5)
                
                print(f"Calling process_chunk_dynamic with num_workers={num_workers}...")
                try:
                    process_chunk_dynamic(api_key, older_convos, 0, len(older_convos),
                                        results_folder, num_workers, book_number)
                    print("process_chunk_dynamic completed successfully for older conversations")
                except Exception as e:
                    print(f"Error in process_chunk_dynamic for older conversations: {e}")
                    traceback.print_exc()
                    if not new_newer_bullets:
                        return False
                
                combined_files = glob.glob(os.path.join(results_folder, "combined_bullets_*.md"))
                print(f"Found {len(combined_files)} combined files after processing older conversations")
                
                if combined_files:
                    older_file = max(combined_files, key=os.path.getctime)
                    older_target = os.path.join(results_folder, "older_bullets.md")
                    print(f"Renaming {older_file} to {older_target}")
                    os.rename(older_file, older_target)
                    
                    with open(older_target, 'r') as f:
                        lines = f.readlines()
                    
                    # Skip header lines if they exist
                    start_idx = 0
                    for i, line in enumerate(lines):
                        if line.strip() == "":
                            start_idx = i + 1
                            break
                    
                    new_older_bullets = "".join(lines[start_idx:])
                    print(f"Extracted {len(new_older_bullets.splitlines())} new bullet points from older conversations")
            else:
                print("No older conversations were successfully extracted.")
        else:
            print("No older conversations requested or available for expansion.")

        if not new_newer_bullets and not new_older_bullets:
            print("No new bullet points were collected.")
            # Clean up temp directories
            shutil.rmtree(results_folder, ignore_errors=True)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False

        total_new = len(new_newer_bullets.splitlines()) + len(new_older_bullets.splitlines())
        print(f"Collected {total_new} new bullet points.")

        # Open the original book file.
        print("Looking for original book file...")
        book_files = glob.glob(os.path.join(book_folder, f"ChatGPT Life Book #{book_number}.md"))
        if not book_files:
            book_files = glob.glob(os.path.join(book_folder, "*.md"))
            if not book_files:
                print(f"No book file found in {book_folder}.")
                # Clean up temp directories
                shutil.rmtree(results_folder, ignore_errors=True)
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
        
        original_book = book_files[0] if len(book_files) == 1 else max(book_files, key=os.path.getctime)
        print(f"Using original book file: {original_book}")
        
        # Read the original book content
        with open(original_book, 'r') as f:
            book_lines = f.readlines()
        
        print(f"Read {len(book_lines)} lines from original book")
        
        # Extract header and book content without duplicating header/metadata
        header_line = book_lines[0].strip() if book_lines else "# ChatGPT Life Book"
        book_content_body = ""
        metadata_line_idx = -1
        
        # Find the metadata line
        for i, line in enumerate(book_lines):
            if "<!-- Book Metadata:" in line:
                metadata_line_idx = i
                break
        
        # Extract the content after header and metadata
        if metadata_line_idx != -1 and metadata_line_idx + 2 < len(book_lines):
            # Skip header, metadata line, and the empty line after it
            book_content_body = "".join(book_lines[metadata_line_idx + 2:])
            print(f"Extracted {len(book_content_body.splitlines())} lines of content after metadata line")
        elif len(book_lines) > 1:
            # If no metadata found, assume everything after header is content
            book_content_body = "".join(book_lines[1:])
            print(f"No metadata line found, extracted {len(book_content_body.splitlines())} lines after header")
        
        # Get titles for first and last conversation
        new_first_title = ""
        if new_first_idx != first_idx and after_convos > 0:
            # Get the title from the first newer conversation we processed
            if newer_convos and len(newer_convos) > 0:
                new_first_title = newer_convos[0].get('title', 'Untitled')
        else:
            new_first_title = current_first

        new_last_title = ""
        if new_last_idx != last_idx and before_convos > 0:
            # Get the title from the last older conversation we processed
            if older_convos and len(older_convos) > 0:
                new_last_title = older_convos[-1].get('title', 'Untitled')
        else:
            new_last_title = current_last
            
        print(f"New book boundaries - newest: \"{new_first_title}\", oldest: \"{new_last_title}\"")

        # Build the new expanded book.
        new_book_content = f"{header_line}\n"
        new_book_content += f"<!-- Book Metadata: first_convo=\"{new_first_title}\" last_convo=\"{new_last_title}\" -->\n\n"
        
        # Add newer bullets at the beginning
        if new_newer_bullets:
            print(f"Adding {len(new_newer_bullets.splitlines())} newer bullet points at the beginning")
            new_book_content += new_newer_bullets
            if not new_newer_bullets.endswith("\n\n"):
                new_book_content += "\n\n" if not new_newer_bullets.endswith("\n") else "\n"
        
        # Add original content (make sure it doesn't start with a blank line if there are newer bullets)
        if book_content_body:
            print(f"Adding {len(book_content_body.splitlines())} lines from original book")
            if book_content_body.startswith("\n") and new_newer_bullets:
                book_content_body = book_content_body.lstrip("\n")
            new_book_content += book_content_body
        
        # Make sure there's appropriate spacing before older bullets
        if not new_book_content.endswith("\n"):
            new_book_content += "\n"
        elif not new_book_content.endswith("\n\n") and new_older_bullets:
            new_book_content += "\n"
        
        # Add older bullets at the end
        if new_older_bullets:
            print(f"Adding {len(new_older_bullets.splitlines())} older bullet points at the end")
            new_book_content += new_older_bullets
        
        # Ensure the file ends with a newline
        if not new_book_content.endswith("\n"):
            new_book_content += "\n"

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        expanded_path = os.path.join(book_folder, f"ChatGPT Life Book #{book_number} (Expanded {timestamp}).md")
        print(f"Writing expanded book to {expanded_path}")
        
        try:
            with open(expanded_path, 'w') as f:
                f.write(new_book_content)
            print(f"Successfully saved expanded book: {expanded_path}")
        except Exception as e:
            print(f"Error saving expanded book: {e}")
            traceback.print_exc()
            # Clean up temp directories
            shutil.rmtree(results_folder, ignore_errors=True)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False

        # Clean up temporary directories
        if os.path.exists(results_folder):
            try:
                shutil.rmtree(results_folder)
                print("Cleaned up temporary results folder.")
            except Exception as e:
                print(f"Error removing results folder: {e}")
        
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print("Cleaned up temporary directory.")
            except Exception as e:
                print(f"Error removing temp directory: {e}")

        print("Expansion completed successfully.")
        return True
    except Exception as e:
        print(f"Unhandled exception in expand_book: {e}")
        traceback.print_exc()
        return False

def list_books(base_output_dir):
    """Lists all books that have been created."""
    book_dirs = glob.glob(os.path.join(base_output_dir, "Book-*"))
    
    if not book_dirs:
        print("No books found.")
        return
    
    print("\n===== Your Books =====")
    for book_dir in sorted(book_dirs):
        book_number = os.path.basename(book_dir).split("-")[1]
        book_files = glob.glob(os.path.join(book_dir, "*.md"))
        
        if book_files:
            book_file = max(book_files, key=os.path.getctime)
            size_kb = os.path.getsize(book_file) / 1024
            created_date = datetime.fromtimestamp(os.path.getctime(book_file))
            
            # Count lines in the book file to estimate bullet points
            bullet_count = 0
            with open(book_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith("-"):
                        bullet_count += 1
            
            print(f"Book #{book_number}: {bullet_count} insights, {size_kb:.1f} KB - Created on {created_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Path: {book_file}")
        else:
            print(f"Book #{book_number}: No book file found")
    
    print("=====================\n")

def show_help():
    """Displays available commands."""
    print("\n" + '='*19 + " Available Commands " + '='*19)
    print("/help                      - Shows all available commands")
    print("/create book <size>        - Creates a new book using <size> recent conversations")
    # Adds more conversations from the past or future to your existing book
    print("/expand book <book #> <past> <future> - **This feature is currently broken but the ETA on it is a few days**") 
    print("/list books                - Shows all the books you've created")
    print("/index convo <name of conversation> - Returns the index of a specific conversation")
    print("/estimate <size>           - Estimates the cost for analyzing <size> conversations")
    print("/quit                      - Exits the program")
    print("============================" + '='*30 + "\n")

def estimate_cost(size):
    """Estimates the Claude API cost for processing a given number of conversations."""
    # Cost calculation: $3 per 500 conversations
    estimated_cost = (size / 500) * 3
    
    print(f"\nEstimated cost for {size} conversations:")
    print(f"${estimated_cost:.2f} in Claude API credits")
    
    # Add some context
    if size <= 100:
        print("(This is a small book with limited insights)")
    elif size <= 500:
        print("(This is a medium-sized book with good coverage)")
    else:
        print("(This is a comprehensive book with extensive insights)")
    print()

def main():
    # Configuration
    INPUT_FILE = "conversations.json" 
    BASE_OUTPUT_DIR = "Books"
    
    # Check if conversations file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please place your ChatGPT export file in the same directory.")
        return
    
    # Count total available conversations
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
            total_available = len(conversations)
    except Exception as e:
        print(f"Error loading conversations: {e}")
        return
    
    # API key management
    api_key = None
    api_key_file = "api_key.txt"
    
    # Try to load API key from file
    if os.path.exists(api_key_file):
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
    
    if not api_key:
        api_key = input("Enter your Anthropic API key: ").strip()
        # Save API key for future use
        save_key = input("Save API key locally for future use? (y/n): ").strip().lower()
        if save_key in ['y', 'yes']:
            with open(api_key_file, 'w') as f:
                f.write(api_key)
    
    # Welcome message
    print("\n" + "="*60)
    print("Welcome to the ChatGPT Life Book Generator!")
    print(f"Using conversations from: {INPUT_FILE} ({total_available} conversations available)")
    print("Type /help to see available commands.")
    print("="*60 + "\n")
    
    # Command loop
    while True:
        try:
            command = input("> ").strip()
            
            if command == "/quit":
                print("Exiting program. Goodbye!")
                break
                
            elif command == "/help":
                show_help()
                
            elif command == "/list books":
                list_books(BASE_OUTPUT_DIR)

            elif command.startswith("/index convo"):
                parts = command.split(" ", 2)  # split into at most 3 parts
                if len(parts) == 3:
                    title_to_find = parts[2].strip()
                    index_convo(INPUT_FILE, title_to_find)
                else:
                    index_convo(INPUT_FILE)

                
            elif command.startswith("/create book "):
                try:
                    size = int(command.split("/create book ")[1])
                    if size <= 0:
                        print("Please specify a positive number of conversations.")
                        continue
                        
                    # Calculate estimated cost
                    estimated_cost = (size / 500) * 3
                    print(f"Estimated cost in Claude credits: ${estimated_cost:.2f}")
                    
                    confirmation = input("Proceed with creating the book? (y/n): ").strip().lower()
                    if confirmation in ["y", "yes"]:
                        print(f"Creating a new book from {size} conversations...")
                        success, book_number = create_book(api_key, size, BASE_OUTPUT_DIR, INPUT_FILE)
                        if success:
                            print(f"Book #{book_number} successfully created!")
                    else:
                        print("Book creation cancelled.")
                        
                except ValueError:
                    print("Please specify a valid number of conversations.")
                                
            elif command.startswith("/expand book "):
                try:
                    parts = command.split("/expand book ")[1].split()
                    if len(parts) != 3:
                        print("Usage: /expand book <book_number> <before> <after>")
                        continue
                    
                    book_num = int(parts[0])
                    before_convos = int(parts[1])
                    after_convos = int(parts[2])
                    
                    if before_convos < 0 or after_convos < 0:
                        print("Please specify non-negative numbers.")
                        continue
                        
                    if before_convos + after_convos == 0:
                        print("Please specify at least one conversation to add.")
                        continue
                        
                    # Calculate estimated cost
                    total_convos = before_convos + after_convos
                    estimated_cost = (total_convos / 500) * 3
                    print(f"Estimated cost in Claude credits: ${estimated_cost:.2f}")
                    
                    confirmation = input(f"Expand Book #{book_num} with {before_convos} earlier and {after_convos} later conversations? (y/n): ").strip().lower()
                    if confirmation in ["y", "yes"]:
                        print(f"Expanding Book #{book_num}...")
                        success = expand_book(api_key, book_num, before_convos, after_convos, BASE_OUTPUT_DIR, INPUT_FILE)
                        if success:
                            print(f"Book #{book_num} expanded successfully!")
                    else:
                        print("Book expansion cancelled.")
                        
                except ValueError:
                    print("Please specify valid numbers.")
                    
            elif command.startswith("/estimate "):
                try:
                    size = int(command.split("/estimate ")[1])
                    if size <= 0:
                        print("Please specify a positive number of conversations.")
                        continue
                        
                    estimate_cost(size)
                    
                except ValueError:
                    print("Please specify a valid number of conversations.")
                    
            else:
                print("Unknown command. Type /help to see available commands.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            print(f"Error: {e}")
                    
if __name__ == "__main__":
    main()
