##Installing arules
#1. Tools
#2. Install Packages
#3. arules in package name
#Installing arulesViz
#1. Tools
#2. Install Packages 
#2. arulesViz in parulesackage name

library("arules")
library("arulesViz")

#Read external Transaction 
#read.transactions(file.choose("\\Path\\"))
#Load the Groceries Dataset
data("Groceries")

#Checking al the transaction
inspect(Groceries)

#See Size of each Transaction
size(head(Groceries, 10))

#Convert transactions into LIST
LIST(head(Groceries ,3))

#See Frequent items
frequentItems <- eclat(Groceries , parameter = list(supp = 0.2 , maxlen = 15))
inspect(frequentItems)
itemFrequencyPlot(Groceries , topN = 10 , type = "absolute")

#Using apriori function to generate number of frequent itemsets and association rules
rules <- apriori(Groceries , parameter = list(supp = 0.001))
inspect(rules[1:5])

#Sorting the rules
sort_rules = sort(rules , by="lift" , decreasing = FALSE)
inspect(sort_rules[1:5])

#Removing redundant rules
redundant_rules <- is.redundant(rules)
summary(redundant_rules)
rules <- rules[!redundant_rules]

#Things that lead to the buying of the whole milk
rules_before_whole_milk <- apriori(Groceries , parameter = list(supp = 0.001 , conf = 0.6) , appearance = list(default = "lhs" , rhs ="whole milk") , control = list(verbose = FALSE))
inspect(rules_before_whole_milk[1:5])

#Things that are bought after or alongwith whole milk
rules_after_whole_milk <- apriori(Groceries , parameter = list(supp = 0.001 , conf = 0.06) , appearance = list(default = "rhs" , lhs ="whole milk") , control = list(verbose = FALSE))
inspect(rules_after_whole_milk[1:5])

#plotting anf visualizing with the help of arulesViz
plot(rules , method="graph" , interactive = TRUE,shading=NA)
