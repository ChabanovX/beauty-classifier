## Гайд по Contribution

Есть 3 вида issues - TASK, BUG, FEATURE. Таска - какое-либо задание, баг - баг, фича - предложение по улучшению проекта

Вот пример `TASK template`:

```md
Part of <paste issue/PR reference>  
<and/or>  
Revealed from <paste issue/PR reference>  
<and/or>  
Required for <paste issues/PRs references>  
<and/or>  
Requires <paste issues/PRs references>  
<and/or>  
Related to <paste issues/PRs references>  

<Remove the lines above if there are no related issues/PRs>




## Background

<Describe the preconditions and the situation which lead to the problem>




## Problem to solve

<Describe the problem to be solved by this task>
```

Пример заполненного описания:

```md
Part of #13 // здесь просто номер issue (github вставит ссылку)
<and/or>  

// далее по аналогии
Revealed from <paste issue/PR reference>   
<and/or>  
Required for <paste issues/PRs references>  
<and/or>  
Requires <paste issues/PRs references>  
<and/or>  
Related to <paste issues/PRs references>  

<Remove the lines above if there are no related issues/PRs>




## Background

Someone built backend and we need to dockerize it




## Problem to solve

Dockerize backend, write `docker-compose.yaml`, `Dockerfile`

```
