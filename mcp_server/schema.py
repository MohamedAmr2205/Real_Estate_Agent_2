class DecomposeSearchArgs(Basemodel):
    model_config=ConfigDict(extra="forbid")

    query: str =Field(...,description="compund question to decompose into sub-questions and search")
    top_k: int=Field(default=3, ge=1,le=10,description="Maximum number of results per sub-question")