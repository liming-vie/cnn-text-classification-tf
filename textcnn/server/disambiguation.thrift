namespace py disambiguation

struct DisABResponse {
	1: i32 ibegin,
	2: i32 iend,	# -1 means no tag found
	3: string tag, # include iend
	4: double score
}

service Disambiguation {
	list<DisABResponse> run(1:string query)
}
