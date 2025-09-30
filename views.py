from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from langchain_core.messages import HumanMessage
from .agent import compiler, config

# below is the django function-based view
@csrf_exempt
def chatUI(request):
    # check if the request method is a POST method
    if request.method == "POST":
        # get the user_query using the request.POST.get user_query method
        user_query = request.POST.get("user_query", "")
        # check if the user_query is empty, if it is, then return a jasonresponse to remind the user that the user_query is empty
        if not user_query:
            return JsonResponse({"error": "No user query provided"}, status=400)
        # create the human_message using the HumanMessage from the langchain_core.messages
        human_message = HumanMessage(content=user_query)
        # set the response from the compiler you imported and add the config dict to store the memory within the thread_id
        response = compiler.invoke({"messages": [human_message]}, config)
        # create the ai_reply and the tool_calls
        ai_reply = ""
        tool_calls = None

        if "messages" in response and len(response["messages"]) > 0:
            ai_msg = response["messages"][-1]
            ai_reply = getattr(ai_msg, "content", "")
            tool_calls = getattr(ai_msg, "tool_calls", None)
        
        return JsonResponse({
            "response": ai_reply,
            "tool_calls": tool_calls,
        })
    else:
        # render the chatUI using the display.html
        return render(request, "display.html")
