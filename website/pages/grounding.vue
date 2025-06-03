<template>
  <div class="min-h-screen bg-gray-200 flex items-center justify-center p-6">
    <div class="w-full max-w-md flex flex-col">
      <!-- Blue Button Header -->
      <div class="flex mb-6">
        <button
            @click="processGrounding"
            :disabled="!selectedImage || !instruction.trim() || isLoading"
            class="flex-1 flex items-center cursor-pointer justify-center px-6 py-3 bg-blue-500 text-white text-lg font-medium rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-md"
        >
          <span v-if="isLoading" class="flex items-center">
            <svg class="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Processing...
          </span>
          <span v-else>Ground Objects</span>
        </button>
      </div>

      <!-- White Card Container -->
      <div class="bg-white rounded-xl shadow-lg p-6 flex flex-col">
        <!-- Instruction Input -->
        <div class="flex flex-col mb-4">
          <label class="text-sm font-medium text-gray-700 mb-2">Instruction</label>
          <input
              v-model="instruction"
              type="text"
              placeholder="e.g., find all cars, locate red objects..."
              class="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              @keyup.enter="processGrounding"
          />
        </div>

        <!-- Image Section -->
        <div class="flex mb-6">
          <!-- File Input -->
          <input
              ref="fileInput"
              type="file"
              accept="image/*"
              @change="handleFileSelect"
              class="hidden"
          />

          <!-- Image Display/Upload Area -->
          <div
              @click="triggerFileInput"
              @dragover.prevent
              @drop.prevent="handleFileDrop"
              :class="[
              'flex-1 rounded-lg overflow-hidden cursor-pointer transition-all hover:bg-gray-50',
              !selectedImage ? 'border-2 border-dashed border-gray-300 hover:border-blue-400 p-2 flex items-center justify-center' : 'flex'
            ]"
              @dragenter="isDragging = true"
              @dragleave="isDragging = false"
          >
            <!-- Image Preview -->
            <div v-if="selectedImage" class="relative flex-1">
              <img
                  :src="selectedImage"
                  alt="Selected image"
                  class="w-full h-64 object-cover rounded-lg"
              />
              <button
                  @click.stop="clearImage"
                  class="absolute top-2 right-2 flex items-center justify-center bg-red-500 text-white rounded-full w-6 h-6 hover:bg-red-600 text-sm"
              >
                ×
              </button>
            </div>

            <!-- Upload Placeholder -->
            <div v-else class="flex flex-col items-center justify-center text-gray-500 py-2 hover:text-gray-700 transition-colors">
              <p class="text-sm">📷 Click to upload image</p>
            </div>
          </div>
        </div>

        <!-- Result Section -->
        <div v-if="resultImage || error" class="flex flex-col">
          <div class="border-t pt-4">
            <!-- Result Image -->
            <div v-if="resultImage" class="flex flex-col">
              <h3 class="text-sm font-medium text-gray-700 mb-2">Result with Bounding Boxes</h3>
              <div class="flex justify-center mb-4">
                <img
                    :src="resultImage"
                    alt="Grounding result"
                    class="max-w-full h-auto rounded-lg border"
                />
              </div>

              <!-- Detection Summary -->
              <div v-if="detectionResults" class="flex flex-col">
                <div class="flex justify-between items-center mb-2">
                  <span class="text-sm font-medium text-gray-700">Detections</span>
                  <span class="text-xs text-gray-500">{{ detectionResults.length }} found</span>
                </div>
                <div class="bg-gray-50 rounded-lg p-3">
                  <div v-for="(detection, index) in detectionResults" :key="index" class="flex justify-between items-center py-1">
                    <span class="text-sm text-gray-700">{{ detection.label }}</span>
                    <span class="text-xs text-gray-500">{{ (detection.score * 100).toFixed(1) }}%</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- Error Display -->
            <div v-else-if="error" class="flex justify-center">
              <div class="text-red-500 text-lg text-center">
                {{ error }}
              </div>
            </div>
          </div>
        </div>

        <!-- Placeholder when no results -->
        <div v-else class="flex justify-center">
          <div class="text-gray-400 text-lg text-center">
            Upload an image, enter instruction, and click "Ground Objects" to find objects
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

// Page metadata
useHead({
  title: 'Visual Grounding',
  meta: [
    { name: 'description', content: 'AI-powered visual grounding service' }
  ]
})

// Reactive state
const fileInput = ref(null)
const selectedImage = ref(null)
const selectedFile = ref(null)
const instruction = ref('')
const resultImage = ref('')
const detectionResults = ref(null)
const isLoading = ref(false)
const error = ref('')
const isDragging = ref(false)

// File handling methods
const triggerFileInput = () => {
  fileInput.value?.click()
}

const handleFileSelect = (event) => {
  const file = event.target.files[0]
  if (file) {
    processFile(file)
  }
}

const handleFileDrop = (event) => {
  isDragging.value = false
  const file = event.dataTransfer.files[0]
  if (file && file.type.startsWith('image/')) {
    processFile(file)
  }
}

const processFile = (file) => {
  // Validate file size (10MB limit)
  if (file.size > 10 * 1024 * 1024) {
    error.value = 'File size must be less than 10MB'
    return
  }

  selectedFile.value = file

  // Create preview URL
  const reader = new FileReader()
  reader.onload = (e) => {
    selectedImage.value = e.target.result
    resultImage.value = ''
    detectionResults.value = null
    error.value = ''
  }
  reader.readAsDataURL(file)
}

const clearImage = () => {
  selectedImage.value = null
  selectedFile.value = null
  resultImage.value = ''
  detectionResults.value = null
  error.value = ''
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

const processGrounding = async () => {
  if (!selectedFile.value) {
    error.value = 'Please select an image first'
    return
  }

  if (!instruction.value.trim()) {
    error.value = 'Please enter an instruction'
    return
  }

  isLoading.value = true
  error.value = ''
  resultImage.value = ''
  detectionResults.value = null

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)

    // Encode instruction as URL parameter
    const encodedInstruction = encodeURIComponent(instruction.value.trim())
    const url = `http://localhost:8000/api/ground-objects?instruction=${encodedInstruction}`

    const response = await fetch(url, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || 'Failed to process grounding request')
    }

    const result = await response.json()
    resultImage.value = result.processed_image
    detectionResults.value = result.detections || []

  } catch (err) {
    error.value = err.message || 'An error occurred while processing the grounding request'
    console.error('Grounding processing error:', err)
  } finally {
    isLoading.value = false
  }
}
</script>