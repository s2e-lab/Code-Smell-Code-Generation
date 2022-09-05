'''
some useful information about memory allocation in operating system

->There are various algorithms which are implemented by the Operating System in order to find out the holes(continuous empy blocks) 
  in the linked list(array in this kata) and allocate them to the processes.

->various algorithms used by operating system:
    1. First Fit Algorithm => First Fit algorithm scans the linked list and whenever it finds the first big enough hole to store a process, it stops scanning and load the process into that hole.
    
    2. Next Fit Algorithm  => Next Fit algorithm is similar to First Fit algorithm except the fact that, Next fit scans the linked list from the node where it previously allocated a hole.
                              ( if i have allocated memory of size 8 in previous turn and initial pointer is 3 
                                then in next turn os will start searching for next empty hole from position 11(3+8=11) )
    
    3. Best Fit Algorithm  => The Best Fit algorithm tries to find out the smallest hole possible in the list that can accommodate the size requirement of the process.
    
    4. Worst Fit Algorithm => it is opposite of Best Fit Algorithm meaning that 
                              (The worst fit algorithm scans the entire list every time and tries to find out the biggest hole in the list which can fulfill the requirement of the process.)
    
    The first fit and best fit algorithms are the best algorithm among all

PS. I HAVE IMPLEMENTED Best Fit Algorithm IN JAVASCRIPT AND IMPLEMENTED Next Fit Algorithm in PYTHON :)
'''

#Next fit Algorithm
class MemoryManager:
    def __init__(self, memory):
        self.storage = [True] * len(memory)
        self.previous_allocated_index = 0
        self.allocated = {}
        self.data = memory

    def allocate(self, size):
        find_next = self.process_allocate(self.previous_allocated_index, len(self.data) - size + 1, size)  # start searching from previously allocated block
        if find_next is not None : return find_next
        from_start = self.process_allocate(0, self.previous_allocated_index - size + 1, size)              # if we cant find from last index then start searching from starting to previously allocated index
        if from_start is not None : return from_start
        raise IndexError('caused by insufficient space in storage')
    
    def process_allocate(self, initial, end, size):
        for i in range(initial, end):  
            if all(self.storage[i:i + size]):
                self.previous_allocated_index = i
                self.storage[i:i + size] = [False] * size
                self.allocated[i] = i + size
                return i
    
    def release(self, pointer):
        if self.storage[pointer] : raise RuntimeError('caused by providing incorrect pointer for releasing memory')
        size = self.allocated[pointer] - pointer
        self.storage[pointer:size] = [True] * size
        self.data[pointer:size] = [None] * size
        del self.allocated[pointer]

    def read(self, pointer):
        if self.storage[pointer] : raise RuntimeError('caused by providing incorrect pointer for reading memory')
        return self.data[pointer]

    def write(self, pointer, value):
        if self.storage[pointer] : raise RuntimeError('caused by providing incorrect pointer for writing memory')
        self.data[pointer] = value
class MemoryManager:
    def __init__(self, memory):
        """
        @constructor Creates a new memory manager for the provided array.
        @param {memory} An array to use as the backing memory.
        """
        self.mem = memory
        self.blockpointers = [] # A list of pointers to the start of each allocated block
        self.blocksizes = []    # A list of sizes of each block
        

    def allocate(self, size):
        """
        Allocates a block of memory of requested size.
        @param {number} size - The size of the block to allocate.
        @returns {number} A pointer which is the index of the first location in the allocated block.
        @raises If it is not possible to allocate a block of the requested size.
        """
        if size > len(self.mem):
            raise Exception("Cannot allocate more memory than exists")
            
        #check start of memory
        if self.blockpointers == [] or (0 not in self.blockpointers and size <= self.blockpointers[0]):
            self.blockpointers.insert(0,0)
            self.blocksizes.insert(0,size)
            return 0
        
        #check after every allocated block
        for i,e in enumerate(self.blocksizes[:-1]):
            if size <= (self.blockpointers[i+1]-self.blockpointers[i]-e):
                self.blockpointers.insert(i,self.blockpointers[i] + e)
                self.blocksizes.insert(i,size)
                return self.blockpointers[i]
                
        #check after last allocated block
        if size <= (len(self.mem) - self.blockpointers[-1] - self.blocksizes[-1]):
            self.blockpointers.append(self.blockpointers[-1] + self.blocksizes[-1])
            self.blocksizes.append(size)
            return self.blockpointers[-1]
            
        raise Exception("Cannot allocate more memory than available")
                
            
    def release(self, pointer):
        """
        Releases a previously allocated block of memory.
        @param {number} pointer - The pointer to the block to release.
        @raises If the pointer does not point to an allocated block.
        """
        if pointer not in self.blockpointers:
            raise Exception("No memory has been allocated")
        
        index = self.blockpointers.index(pointer)
        self.blockpointers.pop(index)
        self.blocksizes.pop(index)
        return

    def read(self, pointer):
        """
        Reads the value at the location identified by pointer
        @param {number} pointer - The location to read.
        @returns {number} The value at that location.
        @raises If pointer is in unallocated memory.
        """
        if not self.inMemory(pointer):
            raise Exception("No memory has been allocated")
        return self.mem[pointer]


    def write(self, pointer, value):
        """
        Writes a value to the location identified by pointer
        @param {number} pointer - The location to write to.
        @param {number} value - The value to write.
        @raises If pointer is in unallocated memory.
        """
        if not self.inMemory(pointer):
            raise Exception("No memory has been allocated")
        self.mem[pointer] = value
        
    def inMemory(self,pointer):    
        """
        Checks if pointer is in allocated memory
        @param {number} pointer - The location in memory.
        @raises If pointer is in unallocated memory.
        """
        i = 0
        while pointer < self.blockpointers[i] + self.blocksizes[i]:
            if pointer >= self.blockpointers[i] and i < self.blockpointers[i] + self.blocksizes[i]:
                return True
            i += 1 
        return False

import copy
class MemoryManager:
    def __init__(self, memory:list):
        """
        @constructor Creates a new memory manager for the provided array.
        @param {memory} An array to use as the backing memory.
        """
        self.memory = memory
        self.Firstindex = copy.copy(memory)
        self.lens = len(self.Firstindex)
    def allocate(self, size):
        """
        Allocates a block of memory of requested size.
        @param {number} size - The size of the block to allocate.
        @returns {number} A pointer which is the index of the first location in the allocated block.
        @raises If it is not possible to allocate a block of the requested size.
        """
        if(size > self.lens):
            raise Exception("allocate size ERROR")
        else:
            # index = 0
            tempindex = self.Firstindex
            while(tempindex):
                if(tempindex.count(None) == 0):
                    # print(tempindex)
                    break
                else:
                    index = self.Firstindex.index(None)
                if(index+size > self.lens):
                    break
                else:
                    last = index+size
                if(self.Firstindex[index:last] == [None]*size):
                    self.Firstindex[index:last] = [index]*size
                    return index
                else:
                    needlist = self.Firstindex[index:last]
                    s2 = list(filter(None, needlist))
                    tempindex = (tempindex[tempindex.index(s2[-1]) + 1:])

            raise Exception("allocate END ERROR")
    def release(self, pointer:int):
        """
        Releases a previously allocated block of memory.
        @param {number} pointer - The pointer to the block to release.
        @raises If the pointer does not point to an allocated block.
        """
        if(pointer not in self.Firstindex):
            raise Exception("pointer release ERROR")
        counts = self.Firstindex.count(pointer)
        first = self.Firstindex.index(pointer)
        last = first +  counts
        self.memory[first:last] = [None]*(counts)
        self.Firstindex[first:last] = [None]*(counts)

    def read(self, pointer):
        """
        Reads the value at the location identified by pointer
        @param {number} pointer - The location to read.
        @returns {number} The value at that location.
        @raises If pointer is in unallocated memory.
        """
        if(pointer >= self.lens):
            raise Exception("pointer read ERROR1")
        if(self.Firstindex[pointer] == None):
            raise Exception("pointer read ERROR2")
        else:
            return self.memory[pointer]
    def write(self, pointer, value):
        """
        Writes a value to the location identified by pointer
        @param {number} pointer - The location to write to.
        @param {number} value - The value to write.
        @raises If pointer is in unallocated memory.
        """
        if (pointer >= self.lens):
            raise Exception()
        if (self.Firstindex[pointer] == None):
            raise Exception()
        else:
            self.memory[pointer] = value
class MemoryManager:
    def __init__(self, memory):
        """
        @constructor Creates a new memory manager for the provided array.
        @param {memory} An array to use as the backing memory.
        """
        self.memory = memory
        self.allocated = {}
        self.free_memory = {0:len(memory)}

    def allocate(self, size):
        """
        Allocates a block of memory of requested size.
        @param {number} size - The size of the block to allocate.
        @returns {number} A pointer which is the index of the first location in the allocated block.
        @raises If it is not possible to allocate a block of the requested size.
        """
        if size > len(self.memory):
            raise 'Cannot allocate more memory than exists'
        for pointer, block_size in list(self.free_memory.items()):
            if block_size >= size:
                self.allocated[pointer] = size
                self.free_memory[pointer+size]= block_size-size
                del self.free_memory[pointer]
                return pointer
        raise 'Cannot allocate more memory than available'
    
    def release(self, pointer):
        """
        Releases a previously allocated block of memory.
        @param {number} pointer - The pointer to the block to release.
        @raises If the pointer does not point to an allocated block.
        """
        self.free_memory[pointer] = self.allocated[pointer]
        del self.allocated[pointer]
        for p, b_size in sorted(self.free_memory.items()):
            if self.free_memory.get(p+b_size):
                self.free_memory[p] += self.free_memory[p+b_size]
                del self.free_memory[p+b_size]

    def read(self, pointer):
        """
        Reads the value at the location identified by pointer
        @param {number} pointer - The location to read.
        @returns {number} The value at that location.
        @raises If pointer is in unallocated memory.
        """
        for p, b_size in list(self.allocated.items()):
            if p <= pointer < p+b_size:
                return self.memory[pointer]
        raise 'No memory has been allocated'

    def write(self, pointer, value):
        """
        Writes a value to the location identified by pointer
        @param {number} pointer - The location to write to.
        @param {number} value - The value to write.
        @raises If pointer is in unallocated memory.
        """
        for p, b_size in list(self.allocated.items()):
            if p <= pointer < p+b_size:
                self.memory[pointer] = value
                return None
        raise 'No memory has been allocated'
            
            

class MemoryManager:
    def __init__(self, memory):
        """
        @constructor Creates a new memory manager for the provided array.
        @param {memory} An array to use as the backing memory.
        """
        self.memory = memory
        self.available_memory = len(memory)
        self.pointer_list = {0:0}
        self.pointer = 0

    def allocate(self, size):
        """
        Allocates a block of memory of requested size.
        @param {number} size - The size of the block to allocate.
        @returns {number} A pointer which is the index of the first location in the allocated block.
        @raises If it is not possible to allocate a block of the requested size.
        """
        if self.available_memory >= size:
            self.pointer_list[self.pointer] = size
            _pointer = self.pointer
            self.pointer += size
            self.available_memory -= size
            return _pointer
        else:
            raise Exception

    def release(self, pointer):
        """
        Releases a previously allocated block of memory.
        @param {number} pointer - The pointer to the block to release.
        @raises If the pointer does not point to an allocated block.
        """
        free_memory = self.pointer_list[pointer]
        self.available_memory += free_memory
#         self.pointer -= free_memory
        self.pointer = pointer if pointer < self.pointer else self.pointer
        del self.pointer_list[pointer]

    def read(self, pointer):
        """
        Reads the value at the location identified by pointer
        @param {number} pointer - The location to read.
        @returns {number} The value at that location.
        @raises If pointer is in unallocated memory.
        """
        for p, memory_block in self.pointer_list.items():
            if pointer <= memory_block:
                return self.memory[pointer]
        raise Exception

    def write(self, pointer, value):
        """
        Writes a value to the location identified by pointer
        @param {number} pointer - The location to write to.
        @param {number} value - The value to write.
        @raises If pointer is in unallocated memory.
        """
        for p, memory_block in self.pointer_list.items():
            if pointer < memory_block:
                self.memory[pointer] = value
                return
        raise Exception
class MemoryManager:
    def __init__(self, memory):
        """
        @constructor Creates a new memory manager for the provided array.
        @param {memory} An array to use as the backing memory.
        """
        self._memory = memory
        self._capacity = len(memory)
        self._allocated = {i: False for i in range(self._capacity)}
        self._pointers = dict() # Holds all current valid pointers, index -> size

    def allocate(self, size):
        """
        Allocates a block of memory of requested size.
        @param {number} size - The size of the block to allocate.
        @returns {number} A pointer which is the index of the first location in the allocated block.
        @raises If it is not possible to allocate a block of the requested size.
        """
        if size > self._capacity:
            raise Exception("Request exceeds max system capacity")

        block_start = self._find_empty_block(size)
        if block_start is None:
            raise Exception("No free block of sufficient size found")

        self._pointers[block_start] = size
        for i in range(size):
            self._allocated[block_start + i] = True

        return block_start


    def release(self, pointer):
        """
        Releases a previously allocated block of memory.
        @param {number} pointer - The pointer to the block to release.
        @raises If the pointer does not point to an allocated block.
        """
        if pointer not in self._pointers:
            raise Exception("Pointer was not allocated")

        pointer_size = self._pointers[pointer]
        for i in range(pointer_size):
            self._allocated[pointer + i] = False

        del self._pointers[pointer]

    def read(self, pointer):
        """
        Reads the value at the location identified by pointer
        @param {number} pointer - The location to read.
        @returns {number} The value at that location.
        @raises If pointer is in unallocated memory.
        """
        if not self._allocated[pointer]:
            raise Exception("Memory space not allocated")

        return self._memory[pointer]


    def write(self, pointer, value):
        """
        Writes a value to the location identified by pointer
        @param {number} pointer - The location to write to.
        @param {number} value - The value to write.
        @raises If pointer is in unallocated memory.
        """
        if not self._allocated[pointer]:
            raise Exception("Cannot write at unallocated memory space")

        self._memory[pointer] = value

    def _find_empty_block(self, size):
        """
        Find an free block of size size
        :param size: Pointer size
        :return: index of block start
        """
        contiguous_size = 0
        index = 0
        start_index = None

        while index < self._capacity:
            if index in self._pointers:
                start_index = None
                size_to_skip = self._pointers[index]
                index = index + size_to_skip
                continue
            else:
                if start_index is None:
                    start_index = index

                contiguous_size += 1

                if contiguous_size == size:
                    return start_index

                index += 1

        return None
# https://www.codewars.com/kata/525f4206b73515bffb000b21/train/javascript
# https://www.codewars.com/kata/51c8e37cee245da6b40000bd/train/python
# https://www.codewars.com/kata/536e7c7fd38523be14000ca2/train/python
# https://www.codewars.com/kata/52b7ed099cdc285c300001cd/train/python

# [0, 0, 0, None, None, None, None, None]
#           
#  u0437u0430u043du044fu0442u044bu0435

# 2

class MemoryManager:
    def __init__(self, memory):
        self.memory = [None] * len(memory)
        self.disk = memory

    def allocate(self, size):
      previous = 0
      for i in range(len(self.memory)):
        if self.memory[i] == None:
          previous += 1
        else:
          previous = 0
        if previous == size:
          start_index = i +1 - size 
          for x in range(start_index, i + 1):
            self.memory[x] = start_index
          return start_index
      raise Exception('No space available')

    
    def release(self, pointer):
      if pointer not in self.memory:
        raise Exception('pointer not in memory')
      for i in range(len(self.memory)):
        if self.memory[i] == pointer:
           self.memory[i] = None

    def read(self, pointer):
        if self.memory[pointer] == None:
          raise Exception('No space alocated')
        return self.disk[pointer]
    def write(self, pointer, value):
      if self.memory[pointer] == None:
        raise Exception('No space alocated')
      self.disk[pointer] = value

class MemoryManager:
    def __init__(self, memory):
        self.memory = [None] * len(memory)
        self.disk = memory

    def allocate(self, size):
      previous = 0
      for i in range(len(self.memory)):
        if self.memory[i] == None:
          previous += 1
        else:
          previous = 0
        if previous == size:
          start_index = i +1 - size 
          for x in range(start_index, i + 1):
            self.memory[x] = start_index
          return start_index
      raise Exception('No space available')

    
    def release(self, pointer):
      if pointer not in self.memory:
        raise Exception('pointer not in memory')
      for i in range(len(self.memory)):
        if self.memory[i] == pointer:
           self.memory[i] = None

    def read(self, pointer):
        if self.memory[pointer] == None:
          raise Exception('No space alocated')
        return self.disk[pointer]
    def write(self, pointer, value):
      if self.memory[pointer] == None:
        raise Exception('No space alocated')
      self.disk[pointer] = value
class MemoryManager:
    
    def __init__(self, memory):
        """
        @constructor Creates a new memory manager for the provided array.
        @param {memory} An array to use as the backing memory.
        """
        self.memory = memory
        self.allocationStarts = []
        self.allocationSizes = []

    def allocate(self, size):
        """
        Allocates a block of memory of requested size.
        @param {number} size - The size of the block to allocate.
        @returns {number} A pointer which is the index of the first location in the allocated block.
        @raises If it is not possible to allocate a block of the requested size.
        """
        for i in range(len(self.memory)):
            ok = True
            for j in range(i, i + size): # j is the currently checking index
                if j >= len(self.memory): # it has reached the end
                    ok = False
                    break
                    raise Exception('No allocation available')
#                 if it's already been allocated
                for k in range(len(self.allocationStarts)):
                    if j >= self.allocationStarts[k] and j < self.allocationStarts[k] + self.allocationSizes[k]:
                        ok = False
                        break
            if ok:
                self.allocationStarts.append(i)
                self.allocationSizes.append(size)
#                 print(i)
                return i
        
        # this shouldn't usually be reached because it would stop when j crosses over the length
        raise Exception('No allocation available')
        
        
    
    def release(self, pointer):
        """
        Releases a previously allocated block of memory.
        @param {number} pointer - The pointer to the block to release.
        @raises If the pointer does not point to an allocated block.
        """
        index = self.allocationStarts.index(pointer) # this raises a value error if it's not found
        del self.allocationStarts[index]
        del self.allocationSizes[index]
        

    def read(self, pointer):
        """
        Reads the value at the location identified by pointer
        @param {number} pointer - The location to read.
        @returns {number} The value at that location.
        @raises If pointer is in unallocated memory.
        """
        if len(self.allocationStarts) == 0:
            raise Exception("No memory has been allocated")
            
        for i in range(len(self.allocationStarts)):
            if not(pointer < self.allocationStarts[i] + self.allocationSizes[i] and pointer >= self.allocationStarts[i]):
                raise Exception("Cannot read from unallocated area")
            
        return self.memory[pointer]

    def write(self, pointer, value):
        """
        Writes a value to the location identified by pointer
        @param {number} pointer - The location to write to.
        @param {number} value - The value to write.
        @raises If pointer is in unallocated memory.
        """
        if len(self.allocationStarts) == 0:
            raise Exception("No memory has been allocated")

        for i in range(len(self.allocationStarts)):
            if not(pointer < self.allocationStarts[i] + self.allocationSizes[i] and pointer >= self.allocationStarts[i]):
                raise Exception("Cannot write to unallocated area")
        
        self.memory[pointer] = value
        
        
        
        
        

class MemoryManager:
    def __init__(self, memory):
        """
        @constructor Creates a new memory manager for the provided array.
        @param {memory} An array to use as the backing memory.
        """
        
        # Key: starting index of block; Value: length of block.
        self.cache = {}
        self.memory = memory

    def allocate(self, size):
        """
        Allocates a block of memory of requested size. Strategy: 
        
        1) assert that size is less than memory size, 
        2) add if memory is empty,
        3) allocate memory before the first index,
        4) in between each block, 
        5) Lastly try after the last index + length
        6) Throw error
        
        @param {number} size - The size of the block to allocate.
        @returns {number} A pointer which is the index of the first location in the allocated block.
        @raises If it is not possible to allocate a block of the requested size.
        """
        
        if size > len(self.memory):
            raise Exception("Cannot allocate mory memory than exists")
        elif len(self.cache.items()) == 0:
            self.cache[0] = size - 1
            return 0
                
        for start, length in self.cache.items():
            index = list(self.cache.keys()).index(start)
            next_block = list(self.cache.items())[index + 1] if index + 1 < len(self.cache.items()) else None
            if index == 0 and (size - 1) < start:
                # This is the first block and there is enough space
                # between the start of the overall memory and the start
                # of the first block.
                self.cache[0] = size - 1
                return 0
            elif next_block and next_block[0] - (start + length + 1) >= size:
                # There is a next block, the difference between the 
                # start of the next block and the end of the current block
                # is large enough to fit the requested block size.
                self.cache[start + length + 1] = size - 1
                return start + length + 1
            elif next_block is None and len(self.memory) - (start + length + 1) >= size:
                # This is the last block and there is enough space
                # between the end of overall memory and the end
                # of the last block.
                self.cache[start + length + 1] = size - 1
                return start + length + 1
            
        raise Exception("Failed to allocate memory")
        
    def release(self, pointer):
        """
        Releases a previously allocated block of memory.
        @param {number} pointer - The pointer to the block to release.
        @raises If the pointer does not point to an allocated block.
        """
        if pointer not in list(self.cache.keys()):
            raise Exception("Pointer does not point towards allocated block")

        del self.cache[pointer]
            
    def read(self, pointer):
        """
        Reads the value at the location identified by pointer
        @param {number} pointer - The location to read.
        @returns {number} The value at that location.
        @raises If pointer is in unallocated memory.
        """
        
        if len(self.cache.items()) == 0:
            raise Exception("No memory has been allocated")
        
        for start, length in self.cache.items():
            if pointer not in range(start, start + length):
                raise Exception("Pointer does not point towards allocated block")
        
        return self.memory[pointer]

    def write(self, pointer, value):
        """
        Writes a value to the location identified by pointer
        @param {number} pointer - The location to write to.
        @param {number} value - The value to write.
        @raises If pointer is in unallocated memory.
        """
        
        if len(self.cache.items()) == 0:
            raise Exception("No memory has been allocated")
        
        for start, length in self.cache.items():
            if pointer not in range(start, start + length + 1):
                raise Exception("Pointer does not point towards allocated block")

        self.memory[pointer] = value
