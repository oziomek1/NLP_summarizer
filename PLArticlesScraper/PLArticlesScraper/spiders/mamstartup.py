# -*- coding: utf-8 -*-
import scrapy


class MamstartupSpider(scrapy.Spider):
    name = 'mamstartup'
    allowed_domains = ['mamstartup.pl']
    start_urls = ['http://mamstartup.pl/']

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS_MAIN = "//div[contains(@id, 'boundary')]//section[contains(@class, 'section')]//div/article/figure/a/@href"
    ARTICLES_URLS_ASIDE = "//div[contains(@id, 'boundary')]//section[contains(@class, 'section')]//aside//article/figure/a/@href"
    SUB_PAGE_URL = "//div[contains(@class, 'submenu')]/a/@href"

    TITLE_NODE = "//article[contains(@class, 'article')]/header/div/h1/text()"
    LEAD_NODE = "//article[contains(@class, 'article')]//div[contains(@class, 'article-lead')]/text()"
    TEXT_NODE = "//article[contains(@class, 'article')]//div[contains(@class, 'article-body')]//p/text()"
    TEXT_LIST_NODE = "//article[contains(@class, 'article')]//div[contains(@class, 'article-body')]//ul/li/text()"

    def parse(self, response):
        # iterate through links to articles
        for quote in response.xpath(self.ARTICLES_URLS_MAIN) + response.xpath(self.ARTICLES_URLS_ASIDE):
            yield response.follow(quote, callback=self.parse_article)
        
        #iterate through sub page
        for next_page in response.xpath(self.SUB_PAGE_URL):
            yield response.follow(next_page, self.parse)

    def parse_article(self, response):
        yield {
            'url': response.url,
            'title': response.xpath(self.TITLE_NODE).extract(),
            'lead': response.xpath(self.LEAD_NODE).extract(),
            'text': response.xpath(self.TEXT_NODE).extract(),
            'text_list': response.xpath(self.TEXT_LIST_NODE).extract(),
        }


