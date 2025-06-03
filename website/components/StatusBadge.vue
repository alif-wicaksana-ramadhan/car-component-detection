<template>
  <span
      :class="badgeClasses"
      class="px-4 py-2 rounded-full text-sm font-medium min-w-[70px] text-center"
  >
    {{ displayText }}
  </span>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  status: {
    type: String,
    default: 'unknown'
  }
})

const badgeClasses = computed(() => {
  switch (props.status?.toLowerCase()) {
    case 'open':
      return 'bg-green-500 text-white'
    case 'closed':
      return 'bg-red-500 text-white'
    default:
      return 'bg-gray-400 text-white'
  }
})

const displayText = computed(() => {
  if (!props.status) return 'Unknown'
  return props.status.charAt(0).toUpperCase() + props.status.slice(1)
})
</script>