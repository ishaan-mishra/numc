# numc

### Provide answers to the following questions.
- How many hours did you spend on the following tasks?
  - Task 1 (Matrix functions in C): 10 to 12 hours
  - Task 2 (Speeding up matrix operations): 18 hours
- Was this project interesting? What was the most interesting aspect about it?
  - The idea of the project was really interesting. I am personally very interested in algorithms and data structures. If I was otherwise asked about the fastest way to do matrix multiplication (that I could actually implement in code), I would say Strassen multiplication. However, seeing here that I can get 86x speedup (and, I've heard, possibly even more) on even smaller matrices, by understanding the architecture of computers themselves, is amazing.
  - The most interesting aspect about the project for me was exploiting the structure of the cache by transposing the right matrix, and adjusting SIMD and loop unrolling accordingly to exploit the cache structure (of spatial and temporal locality) using the knowledge that data in cache is stored in cache lines.
- What did you learn?
  - I believe that I learned a lot about almost every concept in the course through this project. Obviously, everything about this project relied on the understanding of how memory is structured under the hood, which is a large part of what we covered in the first few weeks in the class, and during caches. I also implemented stuff to exploit the structure of the datapath of a processor (like loop unrolling to avoid branching hazards and dependencies).
- Is there anything you would change?
  - Not really.
