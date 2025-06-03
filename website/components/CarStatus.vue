<template>
  <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-8 max-w-lg mx-auto">
    <h2 class="text-2xl font-medium text-gray-800 text-center mb-8">Car Status</h2>

    <div class="space-y-6">
      <!-- Front Right Door -->
      <div class="flex items-center justify-between">
        <span class="text-gray-600 font-normal text-lg">Front Right</span>
        <StatusBadge :status="predictions?.front_right_door" />
      </div>

      <!-- Front Left Door -->
      <div class="flex items-center justify-between">
        <span class="text-gray-600 font-normal text-lg">Front Left</span>
        <StatusBadge :status="predictions?.front_left_door" />
      </div>

      <!-- Rear Right Door -->
      <div class="flex items-center justify-between">
        <span class="text-gray-600 font-normal text-lg">Rear Right</span>
        <StatusBadge :status="predictions?.rear_right_door" />
      </div>

      <!-- Rear Left Door -->
      <div class="flex items-center justify-between">
        <span class="text-gray-600 font-normal text-lg">Rear Left</span>
        <StatusBadge :status="predictions?.rear_left_door" />
      </div>

      <!-- Hood -->
      <div class="flex items-center justify-between">
        <span class="text-gray-600 font-normal text-lg">Hood</span>
        <StatusBadge :status="predictions?.hood" />
      </div>
    </div>

    <!-- Last Updated -->
    <div v-if="lastUpdated" class="mt-6 text-center text-sm text-gray-400">
      Last updated: {{ formatTime(lastUpdated) }}
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  predictions: {
    type: Object,
    default: () => ({})
  },
  lastUpdated: {
    type: Number,
    default: null
  }
})

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  return new Date(timestamp * 1000).toLocaleTimeString()
}
</script>