import json
import os
import glob
from datetime import datetime
import anthropic
import time
import random
import threading
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
            
            print(f"Worker {self.worker_id}: Saved {len(all_bullets)} raw bullet points to {raw_bullets_file}")

def worker_task(task_queue, output_dir, worker_id, api_key, shared_counter, total_tasks, print_lock):
    """Worker that pulls tasks from a shared queue and updates a shared counter.
    The worker no longer prints progress updates - that's handled by a separate thread."""
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
        with open(raw_bullets_file, 'w', encoding='utf-8') as f:
            f.write(f"# Raw Data Points - Worker {worker_id}\n\n")
            for bullet in local_bullets:
                f.write(f"{bullet}\n")

def combine_results(output_dir, chunk_info=""):
    """Combine results from all workers into a single file, strictly within this directory only."""
    # Find all raw bullet point files in this directory ONLY (not subdirectories)
    raw_files = glob.glob(os.path.join(output_dir, "raw_bullet_points_worker*.md"))
    
    if not raw_files:
        print(f"No worker result files found in {output_dir}.")
        return 0
    
    # Print files being combined for debugging
    #print(f"Combining {len(raw_files)} files from {output_dir}:")
    #for file in raw_files:
    #    print(f"  - {os.path.basename(file)}")
    
    # Combine all bullet points
    all_bullets = []
    for file in raw_files:
        with open(file, 'r', encoding='utf-8') as f:
            # Skip the header line
            lines = f.readlines()[2:]
            file_bullets = [line.strip() for line in lines if line.strip()]
            all_bullets.extend(file_bullets)
            # print(f"  Added {len(file_bullets)} bullets from {os.path.basename(file)}")
    
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
    # Check for existing Book folders - fixing the pattern to match Book-X
    book_dirs = glob.glob(os.path.join(base_dir, "Book-*"))
    
    # Find the highest book number
    highest_book = 0
    for dir in book_dirs:
        try:
            # Extract the book number from the folder name using the correct pattern
            book_num = int(os.path.basename(dir).split("-")[1])
            highest_book = max(highest_book, book_num)
        except (ValueError, IndexError):
            # If the folder doesn't match our naming pattern, skip it
            pass
    
    # Return next book number
    return highest_book + 1

def process_chunk_dynamic(api_key, conversations, chunk_start, chunk_size, chunk_dir, num_workers):
    """Processes all conversations using a shared task queue, counter, and a global progress monitor."""
    # Start timing the process
    start_time = time.time()
    
    chunk_end = min(chunk_start + chunk_size, len(conversations))
    actual_size = chunk_end - chunk_start
    chunk_info = f"Book_{book_number}"  # Updated to use book number instead of chunk

    print(f"\n{'='*50}")
    print(f"Generating a book from {chunk_end} of {len(conversations)} conversations")
    print(f"{'='*50}")
    print(f"Deploying {num_workers} claude agents...")
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
    
    # Make sure progress thread finishes (it should already exit when counter reaches total)
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
    print(f"Book completed in {time_taken_str} with {total_bullets} bullet points.")
    return total_bullets

def organize_bullets_by_time(bullets_file_path, output_file_path):
    """Organize bullets by time period in the final book."""
    import re
    from collections import defaultdict
    
    # Read all bullet points
    with open(bullets_file_path, 'r', encoding='utf-8') as f:
        # Skip header line
        lines = f.readlines()[2:]  # Skip the first two lines (header and blank line)
        bullet_points = [line.strip() for line in lines if line.strip()]
    
    # Extract date patterns
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
        r'(\d{2}/\d{2}/\d{2})',  # MM/DD/YY
        r'\((\d{4})\)',          # Year in parentheses
        r'\(([A-Za-z]+ \d{4})\)',  # Month YYYY in parentheses
    ]
    
    # Sort bullets by time period
    time_periods = defaultdict(list)
    
    for bullet in bullet_points:
        # Try to extract date
        period = "Unknown Period"
        
        # Look for dates in parentheses at the end or anywhere in the bullet
        for pattern in date_patterns:
            match = re.search(pattern, bullet)
            if match:
                date_str = match.group(1)
                # Try to extract year
                year_match = re.search(r'20\d{2}', date_str)
                if year_match:
                    period = year_match.group(0)
                    break
        
        # Add to appropriate time period
        time_periods[period].append(bullet)
    
    # Sort periods chronologically
    sorted_periods = sorted([p for p in time_periods.keys() if p != "Unknown Period"])
    if "Unknown Period" in time_periods:
        sorted_periods.append("Unknown Period")
    
    # Write organized content
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("# Life Book\n\n")
        f.write("*Generated from ChatGPT conversations*\n\n")
        
        # Write a table of contents
        f.write("## Table of Contents\n\n")
        for idx, period in enumerate(sorted_periods):
            f.write(f"{idx+1}. [{period}](#{period.lower().replace(' ', '-')})\n")
        f.write("\n\n")
        
        # Write introduction
        f.write("## Introduction\n\n")
        f.write("This book contains notable information extracted from my ChatGPT conversations. ")
        f.write("The content is organized chronologically by year where possible, ")
        f.write("with items of unknown date at the end.\n\n")
        
        # Write chapters for each time period
        for period in sorted_periods:
            f.write(f"## {period}\n\n")
            
            # Add bullets for this period
            for bullet in time_periods[period]:
                f.write(f"{bullet}\n\n")  # Add extra newline for better readability
    
    # print(f"Organized {len(bullet_points)} bullet points into {len(sorted_periods)} time periods")
    # print(f"Book saved to: {output_file_path}")
    return len(sorted_periods)

def remove_directory(directory_path):
    """Remove a directory and all its contents."""
    import shutil
    try:
        shutil.rmtree(directory_path)
        return True
    except Exception as e:
        print(f"Error removing directory {directory_path}: {e}")
        return False

if __name__ == "__main__":
    API_KEY = input("Enter your Anthropic API key: ").strip()
    try:
        num_convos_to_use = int(input("Enter the number of conversations to use for your book (recommended 500): ").strip())
    except ValueError:
        num_convos_to_use = 500

    # Estimate costs ($3 per 500 convos): 
    import math
    conversation_batches = math.ceil(num_convos_to_use / 500)
    estimated_cost = (num_convos_to_use / 500) * 3
    
    # Print estimated cost with friendly formatting
    print(f"Estimated cost in Claude Credits: ${estimated_cost:.2f}")
    
    # Ask for confirmation with clear options
    while True:
        confirmation = input("Proceed with creating the book? (y/n): ").strip().lower()
        if confirmation in ["y", "yes"]:
            break
        elif confirmation in ["n", "no"]:
            print("Operation cancelled by user.")
            exit()
        else:
            print("Please enter 'y' or 'n'.")

    INPUT_FILE = "conversations.json"
    BASE_OUTPUT_DIR = "Books"
    NUM_WORKERS = min(10, num_convos_to_use)  # Don't create more workers than conversations
    START_FROM = 0

    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)

    book_number = get_next_book_number(BASE_OUTPUT_DIR)
    book_folder = os.path.join(BASE_OUTPUT_DIR, f"Book-{book_number}")
    os.makedirs(book_folder, exist_ok=True)
    results_folder = os.path.join(book_folder, "Results")
    os.makedirs(results_folder, exist_ok=True)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    total_available = len(conversations)
    if num_convos_to_use > total_available:
        print(f"Only {total_available} conversations available. Using all.")
        num_convos_to_use = total_available
        
        # Recalculate cost with the actual number
        conversation_batches = math.ceil(num_convos_to_use / 500)
        actual_cost = conversation_batches * 3
        if actual_cost < estimated_cost:
            print(f"Adjusted cost: ${actual_cost:.2f} (lower than estimate)")

    # Process the conversations
    total_bullets_collected = process_chunk_dynamic(
        API_KEY, conversations, START_FROM, num_convos_to_use,
        results_folder, NUM_WORKERS
    )

    # Find the combined bullets file and copy it to the book folder
    combined_files = glob.glob(os.path.join(results_folder, "combined_bullets_*.md"))
    book_created = False

    # REPLACE THE EXISTING if combined_files: BLOCK WITH THIS:
    if combined_files:
        newest_file = max(combined_files, key=os.path.getctime)
        book_path = os.path.join(book_folder, f"ChatGPT Life Book #{book_number}.md")
        
        # Simply copy the combined file as the final book with minimal formatting
        with open(newest_file, 'r', encoding='utf-8') as source:
            with open(book_path, 'w', encoding='utf-8') as dest:
                dest.write(f"# ChatGPT Life Book #{book_number}\n\n")
                
                # Skip the first two lines (header and blank line) from the source
                lines = source.readlines()[2:]
                for line in lines:
                    dest.write(f"{line.strip()}\n")  # Remove extra spaces and add single newline
        
        book_created = True
        print(f"Created book with {total_bullets_collected} bullet points")

    import shutil
    if os.path.exists(results_folder):
        try:
            shutil.rmtree(results_folder)
            print(f"Cleaned up temporary files.")
        except Exception as e:
            print(f"Error removing results folder: {e}")
