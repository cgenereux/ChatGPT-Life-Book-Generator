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
                
                # Add metadata with conversation titles instead of indices
                dest.write(f"<!-- Book Metadata: first_convo=\"{first_convo_title}\" last_convo=\"{last_convo_title}\" -->\n\n")
                
                # Skip the first two lines (header and blank line) from the source
                lines = source.readlines()[2:]
                for line in lines:
                    dest.write(f"{line.strip()}\n")
        
        # Also save a metadata file with the conversation titles
        metadata_path = os.path.join(book_folder, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "first_conversation": first_convo_title,
                "last_conversation": last_convo_title,
                "creation_date": datetime.now().isoformat(),
                "bullets": total_bullets_collected
            }, f)

        book_created = True
        print(f"Created book with {total_bullets_collected} bullet points")
        print(f"Book saved to: {book_path}")

        return book_created, book_number

def expand_book(api_key, book_number, before_convos, after_convos, base_output_dir, input_file): 
    """Expands an existing book by adding conversations before and after based on titles."""
    # Check if the book exists
    book_folder = os.path.join(base_output_dir, f"Book-{book_number}")
    if not os.path.exists(book_folder):
        print(f"Book #{book_number} not found.")
        return False
    
    # Try to load metadata
    metadata_path = os.path.join(book_folder, "metadata.json")
    first_conversation = None
    last_conversation = None
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        first_conversation = metadata.get("first_conversation")
        last_conversation = metadata.get("last_conversation")
    else:
        # Try to extract metadata from book file
        book_files = glob.glob(os.path.join(book_folder, "*.md"))
        if not book_files:
            print(f"No book file found in {book_folder}.")
            return False
            
        book_file = max(book_files, key=os.path.getctime)
        with open(book_file, 'r') as f:
            for line in f:
                if "<!-- Book Metadata:" in line:
                    # Extract conversation titles from comment
                    first_match = re.search(r'first_convo="([^"]+)"', line)
                    last_match = re.search(r'last_convo="([^"]+)"', line)
                    if first_match and last_match:
                        first_conversation = first_match.group(1)
                        last_conversation = last_match.group(1)
                        break
            else:
                print("Could not find book metadata. Cannot expand this book.")
                return False
    
    if not first_conversation or not last_conversation:
        print("Incomplete metadata. Cannot expand this book.")
        return False
    
    # Load conversations
    with open(input_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    # Find indices of first and last conversations by title
    first_idx = -1
    last_idx = -1

    for i, convo in enumerate(conversations):
        title = convo.get('title', 'Untitled')
        if title == first_conversation and first_idx == -1:
            first_idx = i
        if title == last_conversation and first_idx != -1:
            last_idx = i
            break

    if first_idx == -1 or last_idx == -1:
        print("Could not find the original conversations in the current file.")
        print(f"Looking for first conversation: \"{first_conversation}\"")
        print(f"Looking for last conversation: \"{last_conversation}\"")
        return False

    # When checking for conversation titles, add more checking
    if first_idx == 0 and first_conversation == conversations[0].get('title', 'Untitled'):
        print("Note: Your book already starts with the first conversation in your JSON file.")
        # Check if we need to shift the "original" range as suggested previously
        if before_convos > 0:
            print(f"Will treat the first {before_convos} conversations as new content.")
            new_first_idx = 0
            first_idx = before_convos  # Shift the "original" range
        
    # For future conversations, add better handling
    if after_convos > 0:
        available_after = len(conversations) - 1 - last_idx
        if available_after == 0:
            print("Your book already includes the last conversation in your JSON file.")
            print("No future conversations available to add.")
        elif available_after < after_convos:
            print(f"Note: Only {available_after} future conversations are available (you requested {after_convos}).")
            print(f"Will add all {available_after} available conversations.")
            after_convos = available_after

    # Calculate new indices
    # Change this:
    # new_first_idx = max(0, first_idx - before_convos)
    # To this: ALWAYS add more conversations from the beginning
    new_first_idx = max(0, first_idx - before_convos)
    if new_first_idx == first_idx and before_convos > 0:
        # If we're at the first conversation (index 0) and user wants more before,
        # just start from 0 and include the original range too
        print(f"Already at the first conversation. Adding the first {before_convos} conversations as new content.")
        new_first_idx = 0
        first_idx = before_convos  # Shift the "original" range to start after the new content
    new_last_idx = min(len(conversations) - 1, last_idx + after_convos)
    
    # Get titles of new first and last conversations
    new_first_title = conversations[new_first_idx].get('title', 'Untitled')
    new_last_title = conversations[new_last_idx].get('title', 'Untitled')
    
    # Create a results folder for temporary files
    results_folder = os.path.join(book_folder, "Results")
    os.makedirs(results_folder, exist_ok=True)
    
    # Process conversations before current range
    before_results = 0
    if before_convos > 0 and new_first_idx < first_idx:
        actual_before = first_idx - new_first_idx
        print(f"Processing {actual_before} earlier conversations...")
        # Adjust worker count for one-sided expansion
        num_workers = min(before_convos, 10)  # Use up to 10 workers, limited by conversation count
        if after_convos > 0:  # If we're doing both directions, limit to 5 for each
            num_workers = min(num_workers, 5)
        before_results = process_chunk_dynamic(
            api_key, conversations, new_first_idx, actual_before,
            results_folder, num_workers, book_number
        )

    # Process conversations after current range
    after_results = 0
    if after_convos > 0 and new_last_idx > last_idx:
        actual_after = new_last_idx - last_idx
        print(f"Processing {actual_after} later conversations...")
        # Adjust worker count for one-sided expansion
        num_workers = min(after_convos, 10)  # Use up to 10 workers, limited by conversation count
        if before_convos > 0:  # If we're doing both directions, limit to 5 for each
            num_workers = min(num_workers, 5)
        after_results = process_chunk_dynamic(
            api_key, conversations, last_idx + 1, actual_after,
            results_folder, num_workers, book_number
    )
    
    total_new_bullets = before_results + after_results
    
    if total_new_bullets > 0:
        # Find the original book file (not an expanded version)
        book_files = glob.glob(os.path.join(book_folder, f"ChatGPT Life Book #{book_number}.md"))
        if not book_files:
            # If original not found, try looking for any file
            book_files = glob.glob(os.path.join(book_folder, "*.md"))
            if not book_files:
                print(f"No book file found in {book_folder}.")
                return False

        # Select the original file or the oldest file if original naming not found        
        original_book = book_files[0] if len(book_files) == 1 else min(book_files, key=os.path.getctime)
        
        # Read original book content (skipping metadata)
        with open(original_book, 'r') as f:
            lines = f.readlines()
            header_line = lines[0]
            content_start = 0
            for i, line in enumerate(lines):
                if "<!-- Book Metadata:" in line:
                    content_start = i + 1
                    break
            original_content = lines[content_start:]
        
        # Find the new combined bullets file
        combined_files = glob.glob(os.path.join(results_folder, "combined_bullets_*.md"))
        new_content = []
        if combined_files:
            newest_file = max(combined_files, key=os.path.getctime)
            with open(newest_file, 'r') as f:
                # Skip the header lines
                for i, line in enumerate(f):
                    if i >= 2:  # Skip first two lines
                        new_content.append(line)
        
        # Create updated book file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        expanded_book_path = os.path.join(book_folder, f"ChatGPT Life Book #{book_number} (Expanded {timestamp}).md")
        with open(expanded_book_path, 'w') as f:
            # Write header
            f.write(header_line)
            f.write("\n")
            # Write updated metadata
            f.write(f"<!-- Book Metadata: first_convo=\"{new_first_title}\" last_convo=\"{new_last_title}\" -->\n\n")
            
            # Write content from before the original range
            before_content = new_content[:before_results]
            for line in before_content:
                f.write(line)
            
            # Write original content
            for line in original_content:
                f.write(line)
            
            # Write content from after the original range
            after_content = new_content[before_results:]
            for line in after_content:
                f.write(line)
        
        # Update metadata file
        with open(metadata_path, 'w') as f:
            json.dump({
                "first_conversation": new_first_title,
                "last_conversation": new_last_title,
                "original_first": first_conversation,
                "original_last": last_conversation,
                "creation_date": metadata.get("creation_date", ""),
                "expansion_date": datetime.now().isoformat(),
                "original_bullets": metadata.get("bullets", 0),
                "added_bullets": total_new_bullets,
                "total_bullets": metadata.get("bullets", 0) + total_new_bullets
            }, f)
        
        print(f"Added {total_new_bullets} new bullet points to the book")
        print(f"Expanded book saved to: {expanded_book_path}")
    else:
        print("No new bullet points were collected.")
    
    # Clean up temporary files
    if os.path.exists(results_folder):
        try:
            shutil.rmtree(results_folder)
            print(f"Cleaned up temporary files.")
        except Exception as e:
            print(f"Error removing results folder: {e}")
    
    return total_new_bullets > 0

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
    print("/expand book <book #> <past> <future> - Adds more conversations from the past or future to your existing book")
    print("/list books                - Shows all the books you've created")
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
        save_key = input("Save API key for future use? (y/n): ").strip().lower()
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
