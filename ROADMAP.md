# Roadmap or Ideas

* Train on `OpenAssistant/oasst1` dataset
* Continuous learning experiment where model notices "surprising" data and then reviews and plays in that space. Generating examples and trying to work through solutions. Update either the model's weights or a LORA?
  * Maybe LORA solve for avoiding losing the original model's knowledge?
* A `bash` oriented model that can use CLI programs such as `bc`, `find`, `grep`, etc. How much of an itelligence lift do we get from tool use?
  * Have an allow-list of commands that the model can use. 
  * Or do inference in a sandboxed environment?