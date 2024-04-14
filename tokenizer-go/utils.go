package main

type Tuple[T any, U any] struct {
	id1 T
	id2 U
}

func get_stats(ids *[]int32, counts *map[Tuple[int32, int32]]int32) *map[Tuple[int32, int32]]int32 {
	if counts == nil {
		counts = new(map[Tuple[int32, int32]]int32)
	}
	for i := 0; i < len((*ids))-1; i++ {
		pair := Tuple[int32, int32]{id1: (*ids)[i], id2: (*ids)[i+1]}
		(*counts)[pair] = (*counts)[pair] + 1
	}
	return counts
}

func merge(ids *[]int32, pair Tuple[int32, int32], idx int32) []int32 {
	new_ids := make([]int32, 0, len((*ids)))

	i := 0
	for i < len((*ids)) {
		if (*ids)[i] == pair.id1 && i < len((*ids))-1 && (*ids)[i+1] == pair.id2 {
			new_ids = append(new_ids, idx)
			i += 2
		} else {
			new_ids = append(new_ids, (*ids)[i])
			i += 1
		}
	}
	return new_ids
}
